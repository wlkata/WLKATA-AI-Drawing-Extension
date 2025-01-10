import math
import platform
import subprocess
import time
import warnings
from typing import Union, Any

import cv2
import serial
import wlkatapython
from matplotlib import pyplot as plt
from wlkatapython import Mirobot_UART


class Drawing:
    def __init__(
            self,
            ser: serial.Serial = None,
            com_port: str = None, baud_rate: int = 115200,
            robot: Mirobot_UART = None,
            min_x: float = 200,
            min_y: float = -50,
            max_x: float = 300,
            max_y: float = 50,
            z: float = None
    ):
        """
        Initializing drawing helper. Will use serial in following order: ser, com_port and baud_rate, robot

        Args:
            ser (:obj:`serial.Serial`, optional): Existing serial
            com_port (:obj:`str`, optional): Serial port (default: None)
            baud_rate (:obj:`int`, optional): Baud rate (default: 115200)
            robot (:obj:`Mirobot_UART`, optional): Existing Wlkata robot instance (default: None)
            min_x (:obj:`float`, optional): Minimum x coordinate (default: None)
            min_y (:obj:`float`, optional): Minimum y coordinate (default: None)
            max_x (:obj:`float`, optional): Maximum x coordinate (default: None)
            max_y (:obj:`float`, optional): Maximum y coordinate (default: None)
            z (:obj:`float`, optional): Z coordinate (default: None)

        Examples:
            >>> serial1 = serial.Serial(port='/dev/tty.usbserial-1120', baudrate=115200)
            >>> drawer1 = Drawing(serial1)

            >>> drawer2 = Drawing(com_port='/dev/tty.usbserial-1120')

            >>> serial1 = serial.Serial(port='COM3', baudrate=115200)
            >>> MT4_1 = wlkatapython.MT4_UART(serial1)
            >>> MT4_1.init(serial1, -1)
            >>> drawer3 = Drawing(robot=MT4_1)

        Raises:
            :class:`Exception`: If serial port or baud_rate is not valid
        """
        self.ser = None
        self.com_port = None
        self.baud_rate = None
        self.robot = None

        if ser is not None:
            assert not ser.is_open, 'Serial is not open'
            self.ser = ser
            self.com_port = ser.port
            self.baud_rate = ser.baudrate
            self.robot = Mirobot_UART()
            self.robot.init(ser, -1)
        elif com_port is not None and baud_rate is not None:
            self.ser = serial.Serial(com_port, baud_rate)
            self.com_port = com_port
            self.baud_rate = baud_rate
            self.robot = Mirobot_UART()
            self.robot.init(ser, -1)
        elif robot is not None:
            self.ser = robot.pSerial
            self.com_port = robot.pSerial.port
            self.baud_rate = robot.pSerial.baudrate
            self.robot = robot
        if self.ser is None:
            warnings.warn("No serial port passed")
            devices = Drawing.detect_ports()
            if devices is None:
                raise Exception("No serial ports detected")
            for i in range(len(devices)):
                print(f'{i}: {devices[i]}')
            selection = input("Which ports would you like to use: ")
            try:
                self.ser = serial.Serial(devices[int(selection)], baud_rate)
                self.com_port = devices[int(selection)]
                self.baud_rate = baud_rate
                self.robot = Mirobot_UART()
                self.robot.init(ser, -1)
                self.robot.pSerial = self.ser
            except:
                # TODO: Error types
                pass
                # raise Exception("Unable to open serial port")

        self.line_segments_groups = None
        self.points_groups = None
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
        self.z = z

    def __del__(self):
        if self.ser:
            self.ser.close()

    @classmethod
    def detect_ports(
            cls
    ) -> list[str]:
        """
        Get available serial ports

        Returns:
            A list of available serial ports
        """
        os_name = platform.system()
        devices = None
        if os_name == 'Darwin':
            try:
                command = "ls /dev/tty.*"
                result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
                if result.returncode == 0:
                    devices = result.stdout.split('\n')[:-1]
            except Exception as e:
                print(f"An error occurred: {e}")
        elif os_name == 'Windows':
            # TODO: Check if is correct
            try:
                # Command to list COM ports
                command = "wmic path Win32_SerialPort get DeviceID,Caption"
                result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)

                if result.returncode == 0:
                    devices = result.stdout.strip().split('\n')[1:]  # Skip the header row
                    devices = [line.strip() for line in devices if line.strip()]  # Remove empty lines
                else:
                    print(f"Error finding COM ports: {result.stderr.strip()}")
            except Exception as e:
                print(f"An error occurred: {e}")
        else:
            pass

        return devices

    def edges_from_image(
            self,
            image_path: str,
            smoothness: float = 0.0001,
            binary_threshold: int = 127,
            method: int = cv2.RETR_TREE,
            blur: bool = True,
            edge_detection_threshold1: int = 100,
            edge_detection_threshold2: int = 200
    ) -> list[list[list[tuple[Any, Any]]]]:

        """
        Extract edges from image and approximate them with line segments based on smoothness.

        Args:
            image_path (:obj:`str`): Input image path.
            smoothness (:obj:`float`, optional): Smoothness parameter. Higher values simplify edges. (default: 0.0001)
            binary_threshold (:obj:`int`, optional): Binary threshold for edge detection. (default: 127)
            method (:obj:`int`, optional): Contour retrieval method. Options:
                    cv2.RETR_TREE (default): Retrieve both inner and outer edges.
                    cv2.RETR_EXTERNAL: Retrieve only the outer edges.
            blur (:obj:`bool`, optional): Whether to apply Gaussian blur before edge detection. (default: True)
            edge_detection_threshold1 (:obj:`int`, optional): Lower threshold for Canny edge detection. (default: 100)
            edge_detection_threshold2 (:obj:`int`, optional): Upper threshold for Canny edge detection. (default: 200)

        Examples:
            Basic usage with a sample image:

            >>> drawer = Drawing(com_port="/dev/tty.usbserial-1120")
            >>> drawer.edges_from_image("image.jpg")

            Specifying all parameters:

            >>> drawer.edges_from_image("image.jpg", smoothness=0.01, binary_threshold=127, method=cv2.RETR_EXTERNAL, blur=False, edge_detection_threshold1=50, edge_detection_threshold2=150)

        Returns:
            List of line segment groups, where each group corresponds to edges in a contour
        """

        # Load the grayscale image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"Failed to load image from path: {image_path}")

        if blur:
            # Apply Gaussian blur
            image = cv2.GaussianBlur(image, (5, 5), 0)

        # Apply binary threshold to the blurred image
        _, binary = cv2.threshold(image, binary_threshold, 255, cv2.THRESH_BINARY_INV)

        # Detect edges on the blurred binary image
        edges_blurred = cv2.Canny(binary, edge_detection_threshold1, edge_detection_threshold2)

        # Find contours from the edges of the blurred image
        contours_blurred, hierarchy = cv2.findContours(edges_blurred, method, cv2.CHAIN_APPROX_SIMPLE)

        # Approximate contours with line segments and organize by groups
        line_segments_groups = []

        for contour in contours_blurred:
            # Approximate the contour with polygons (line segments)
            epsilon = smoothness * cv2.arcLength(contour, True)  # Smoothness controls approximation
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Collect line segments for the current contour
            line_segments = []
            for i in range(len(approx)):
                pt1 = (approx[i][0][0], approx[i][0][1])
                pt2 = (approx[(i + 1) % len(approx)][0][0],
                       approx[(i + 1) % len(approx)][0][1])  # Loop to connect last to first
                line_segments.append([pt1, pt2])

            # Add the current group of line segments to the result
            line_segments_groups.append(line_segments)

        self.line_segments_groups = line_segments_groups
        return line_segments_groups

    def extract_points_groups(
            self,
            line_segments_groups: list[list[tuple[int, int]]] = None
    ) -> list[list[tuple[int, int]]]:
        """
        Extract end points from list of edge segments

        Args:
            line_segments_groups (list[list[tuple[int, int]]], optional): List of line segments groups. (default: None)

        Returns:
            List of points groups, where each group corresponds to points in a contour
        """
        if line_segments_groups is None:
            line_segments_groups = self.line_segments_groups

        points = []

        for line_segments in line_segments_groups:
            group: list[tuple[int, int]] = []
            for segment in line_segments:
                group.append(segment[0])
            group.append((group[0][0], group[0][1]))
            points.append(group)

        self.points_groups = points
        return points

    def invert_point_groups(
            self,
            groups: list[list[tuple[Any, Any]]] = None,
            rotation: int = 0
    ) -> list[list[tuple[Any, Any]]]:
        """
        Invert points to make it view correctly

        Args:
            groups (list[list[tuple[Any, Any]]], optional): List of points groups. (default: None)
            rotation (:obj:`int`, optional): Number of 90 degree clockwise rotations. (default: 0)

        Returns:
            List of points groups, where each group corresponds to points in a contour after inverting
        """
        if groups is None:
            groups = self.points_groups

        result = []
        for group in groups:
            temp_group = []
            for point in group:
                temp_point = [point[0], -point[1]]
                for _ in range(rotation):
                    temp_point[0], temp_point[1] = -temp_point[1], temp_point[0]
                temp_group.append((temp_point[0], temp_point[1]))
            result.append(temp_group)

        self.points_groups = result
        return result

    def resize_point_groups(
            self,
            groups: list[list[tuple[Any, Any]]] = None,
            keep_aspect_ratio: bool = True,
            border: Union[float, tuple[float, float]] = 5
    ) -> list[list[tuple[float, float]]]:
        """
        Resize points to make it view correctly

        Args:
            groups (list[list[tuple[Any, Any]]], optional): List of points groups. (default: None)
            keep_aspect_ratio (:obj:`bool`, optional): Whether to keep aspect ratio. (default: True)
            border (:obj:`Union[float, tuple[float, float]]`, optional): Border size. (default: 5)

        Returns:
            List of points groups, where each group corresponds to points in a contour after resizing
        """
        if type(border) is not tuple:
            border = (border, border)

        temp_min_x = float('inf')
        temp_min_y = float('inf')
        temp_max_x = float('-inf')
        temp_max_y = float('-inf')

        if groups is None:
            # TODO: Remove repeated line segments
            groups = self.points_groups

        for points in groups:
            for point in points:
                temp_min_x = min(temp_min_x, point[0])
                temp_min_y = min(temp_min_y, point[1])
                temp_max_x = max(temp_max_x, point[0])
                temp_max_y = max(temp_max_y, point[1])

        x_bound = self.max_x - self.min_x - 2 * border[0]
        y_bound = self.max_y - self.min_y - 2 * border[1]

        x_range = temp_max_x - temp_min_x
        y_range = temp_max_y - temp_min_y

        if keep_aspect_ratio:
            x_ratio = y_ratio = min(x_bound / x_range, y_bound / y_range)
            x_offset = (x_bound - x_range * x_ratio) / 2 + border[0]
            y_offset = (y_bound - y_range * y_ratio) / 2 + border[1]
        else:
            x_ratio, y_ratio = x_bound / x_range, y_bound / y_range
            x_offset, y_offset = border

        resize_points = []
        for group in groups:
            tmp_group = []
            for point in group:
                x = (point[0] - temp_min_x) * x_ratio + self.min_x + x_offset
                y = (point[1] - temp_min_y) * y_ratio + self.min_y + y_offset
                tmp_group.append((x, y))
            resize_points.append(tmp_group)

        self.points_groups = resize_points
        return resize_points

    def wait_idle(
            self,
            interval: float = 0.5
    ) -> None:
        """
        Wait until robotic arm in Idle state

        Note:
            During the robotic arm homing process, the robotic arm will remain temporarily idle, which might cause it to exit the loop.

        Args:
            interval (:obj:`float`, optional): Time in seconds to wait between idle states. (default: 0.5)

        Returns:
            None
        """
        while self.robot.getState() != "Idle":
            time.sleep(interval)

    def cali_z(
            self,
            homing: bool = True
    ) -> float:
        """
        Manually set the z position for drawing

        Args:
            homing (:obj:`bool`, optional): Whether homing the robotic arm or not. (default: True)

        Returns:
            Z position
        """
        if homing:
            self.robot.homing()
        # TODO: Include wait_idle function in the library
        self.wait_idle(interval=0.5)
        # TODO: Rewrite getcoordinate if possible
        # TODO: getcoordinate return string instead of number
        temp_z = float(self.robot.getcoordinate(3))
        # TODO: Rewrite writecoordinate function with optional input
        self.robot.sendMsg(f"M20 G90 G00 X{self.min_x} Y{self.min_y} Z{temp_z}")
        while (line := input("Moving down step size (\'Q\' for quit): ").lower()) != "q":
            try:
                line = -float(line)
            except ValueError:
                line = 0
            temp_z += line
            print(line)
            self.robot.sendMsg(f"M20 G91 G00 Z{line}")
            self.wait_idle()
        self.z = temp_z
        return self.z

    def draw_area(
            self,
            z_offset: float = 10
    ) -> None:
        """
        Draw the boundary of drawing area

        Args:
            z_offset (:obj:`float`, optional): Z offset when draw the area. (default: 10)

        Returns:
            None
        """
        self.robot.sendMsg(f'M20 G90 G00 X{self.min_x} Y{self.min_y} Z{self.z + z_offset}')
        self.robot.sendMsg(f'M20 G90 G00 X{self.max_x} Y{self.min_y} Z{self.z + z_offset}')
        self.robot.sendMsg(f'M20 G90 G00 X{self.max_x} Y{self.max_y} Z{self.z + z_offset}')
        self.robot.sendMsg(f'M20 G90 G00 X{self.min_x} Y{self.max_y} Z{self.z + z_offset}')
        self.robot.sendMsg(f'M20 G90 G00 X{self.min_x} Y{self.min_y} Z{self.z + z_offset}')

    def preview(self) -> None:
        """
        Preview drawing trajectory

        Returns:
            None
        """
        plt.figure(figsize=(10, 10))

        for group in self.points_groups:
            for i in range(len(group)):
                pt1 = group[i]
                pt2 = group[(i + 1) % len(group)]
                plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color='blue')

        plt.title(f"Approximated Line Segments")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.xlim(self.min_x, self.max_x)
        plt.ylim(self.min_y, self.max_y)
        plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
        plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
        plt.grid(True)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()


    def draw(
            self,
            image_path: str,
            smoothness: float = 0.0001,
            binary_threshold: int = 127,
            method: int = cv2.RETR_TREE,
            blur: bool = True,
            edge_detection_threshold1: int = 100,
            edge_detection_threshold2: int = 200,
            rotation: int = 0,
            keep_aspect_ratio: bool = True,
            border: Union[float, tuple[float, float]] = 5,
    ) -> None:
        """
        Draw the edges of given image

        Args:
            image_path (:obj:`str`): Input image path.
            smoothness (:obj:`float`, optional): Smoothness parameter. Higher values simplify edges. (default: 0.0001)
            binary_threshold (:obj:`int`, optional): Binary threshold for edge detection. (default: 127)
            method (:obj:`int`, optional): Contour retrieval method. Options:
                    cv2.RETR_TREE (default): Retrieve both inner and outer edges.
                    cv2.RETR_EXTERNAL: Retrieve only the outer edges.
            blur (:obj:`bool`, optional): Whether to apply Gaussian blur before edge detection. (default: True)
            edge_detection_threshold1 (:obj:`int`, optional): Lower threshold for Canny edge detection. (default: 100)
            edge_detection_threshold2 (:obj:`int`, optional): Upper threshold for Canny edge detection. (default: 200)
            rotation (:obj:`int`, optional): Number of 90 degree clockwise rotations. (default: 0)
            keep_aspect_ratio (:obj:`bool`, optional): Whether to keep aspect ratio. (default: True)
            border (:obj:`Union[float, tuple[float, float]]`, optional): Border size. (default: 5)

        Returns:
            None
        """
        self.edges_from_image(
            image_path,
            smoothness=smoothness,
            binary_threshold=binary_threshold,
            method=method, blur=blur,
            edge_detection_threshold1=edge_detection_threshold1,
            edge_detection_threshold2=edge_detection_threshold2
        )

        self.extract_points_groups()
        self.invert_point_groups(rotation=rotation)
        self.resize_point_groups(keep_aspect_ratio=keep_aspect_ratio, border=border)

        if not self.pre_drawing():
            return

        for resized_points in self.points_groups:
            self.draw_points(resized_points)


    def draw_points(
            self,
            points: list[tuple[float, float]],
            z_offset: float = 10
    ) -> None:
        """
        Draw given group of points

        Args:
            points (:obj:`list[tuple[float, float]]`): list of points coordinates
            z_offset (:obj:`float`, optional): Z offset when moving between line segments. (default: 10)

        Returns:
            None
        """
        if len(points) == 0:
            return

        try:
            first_point = points[0]
            last_point = points[-1]

            self.robot.sendMsg(f'M20 G90 G00 X{first_point[0]:2f} Y{first_point[1]:2f} Z{self.z + z_offset:2f}')

            for point in points:
                self.robot.sendMsg(f'M20 G90 G00 X{point[0]:2f} Y{point[1]:2f} Z{self.z:2f}')

            self.robot.sendMsg(f'M20 G90 G00 X{last_point[0]:2f} Y{last_point[1]:2f} Z{self.z + z_offset:2f}')
        except KeyboardInterrupt:
            # TODO: Emergency stop
            pass

    @classmethod
    def mid_points(
            cls,
            points: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        """
        Find mid-points of given group of points

        Args:
            points (:obj:`list[tuple[float, float]]`): list of points coordinates

        Returns:
            List of mid-points coordinates
        """
        if len(points) == 0:
            return []

        tmp = []
        for i in range(len(points) - 1):
            pt_x, pt_y = points[i][0] + points[i + 1][0], points[i][1] + points[i + 1][1]
            tmp.append((pt_x / 2, pt_y / 2))

        if points[0] == points[-1]:
            tmp.append((tmp[0][0], tmp[0][1]))

        return tmp

    @classmethod
    def generate_polygon_vertices(
            cls,
            num_side: int
    ) -> list[tuple[float, float]]:
        """
        Generate the list of polygon vertices of given number of sides

        Args:
            num_side (:obj:`int`): Number of sides

        Returns:
            List of polygon vertices closed loop (points[0] == points[-1])
        """
        angle = 2 * math.pi / num_side
        tmp = []
        for i in range(num_side):
            tmp.append((math.cos(i * angle), math.sin(i * angle)))
        tmp.append((tmp[0][0], tmp[0][1]))
        return tmp

    def pre_drawing(self) -> bool:
        """
        Drawing preparation

        Returns:
            True if preparation pass; False otherwise
        """
        self.preview()
        while (line := input('Sample plot looks good? (y/[n]): ')).lower() != 'y':
            if line == '' or line.lower() == 'n':
                # TODO: smoothness
                return False
        self.draw_area()
        if input('Start drawing? (y/[n]): ').lower() != 'y':
            return False
        return True

    def draw_poly(
            self,
            num_side: int,
            depth: int
    ) -> None:
        """
        Draw a polygon with a specified number of sides and iteratively refines it for a given depth.

        Args:
            num_side (:obj:`int`): Number of sides for the base polygon.
            depth (:obj:`int`): Number of iterations to refine and draw the polygon.

        Returns:
            None
        """
        points = Drawing.generate_polygon_vertices(num_side)
        self.points_groups = [points]
        points = self.resize_point_groups(keep_aspect_ratio=True)[0]

        if not self.pre_drawing():
            return

        for _ in range(depth):
            self.points_groups = [points]
            self.draw_points(points)
            points = Drawing.mid_points(points)
            self.wait_idle()


if __name__ == "__main__":
    drawer = Drawing()
    drawer.cali_z()

    # Sample of drawing spiral
    drawer.draw('sample/spiral.jpg')

    # Sample of drawing polygon
    drawer.draw_poly(num_side=5, depth=3)
