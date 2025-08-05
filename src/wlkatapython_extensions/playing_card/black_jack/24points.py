from wlkatapython_extensions.drawing import Drawing

import serial
from wlkatapython import MT4_UART

import cv2
from ultralytics import YOLO
from PIL import Image

import itertools


def simplify_by_eval(expr, target=24, tol=1e-6):
    def eval_expr_safe(s):
        try:
            return eval(s)
        except ZeroDivisionError:
            return None
        except Exception:
            return None

    def find_innermost_pair(s):
        stack = []
        for i, char in enumerate(s):
            if char == '(':
                stack.append(i)
            elif char == ')':
                if stack:
                    start = stack.pop()
                    return start, i  # return the first innermost pair
        return None

    original_val = eval_expr_safe(expr)
    if original_val is None:
        return expr

    while True:
        pair = find_innermost_pair(expr)
        if not pair:
            break
        start, end = pair
        without = expr[:start] + expr[start+1:end] + expr[end+1:]
        val = eval_expr_safe(without)
        if val is not None and abs(val - target) < tol:
            expr = without  # remove parens and restart
        else:
            # can't remove this one — skip it by replacing with placeholder to avoid infinite loop
            expr = expr[:start] + '[' + expr[start+1:end] + ']' + expr[end+1:]

    # restore any skipped ones
    expr = expr.replace('[', '(').replace(']', ')')
    return expr


def solve24(nums):
    ops = ['+', '-', '*', '/']
    for p in itertools.permutations(nums):
        for op1 in ops:
            for op2 in ops:
                for op3 in ops:
                    try:
                        exprs = [
                            f'({p[0]}{op1}{p[1]}){op2}({p[2]}{op3}{p[3]})',
                            f'(({p[0]}{op1}{p[1]}){op2}{p[2]}){op3}{p[3]}',
                            f'({p[0]}{op1}({p[1]}{op2}{p[2]})){op3}{p[3]}',
                            f'{p[0]}{op1}(({p[1]}{op2}{p[2]}){op3}{p[3]})',
                            f'{p[0]}{op1}({p[1]}{op2}({p[2]}{op3}{p[3]}))'
                        ]
                        for expr in exprs:
                            if abs(eval(expr) - 24) < 1e-6:
                                return simplify_by_eval(expr)
                    except ZeroDivisionError:
                        continue
    return None


def capture_and_detect(model: YOLO):
    """
    Captures an image from the camera and processes it using a YOLO model.
    Returns the detected results.
    """

    # Open the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None

    # Capture a single frame
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Failed to capture an image.")
        return None

    # Convert to RGB for YOLO compatibility
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to PIL Image
    img_pil = Image.fromarray(frame_rgb)

    # Run YOLO prediction
    results = model(img_pil, device='mps')

    return results[0].boxes.cls.cpu().to(int).tolist()


def draw_digit(
        drawer: Drawing,
        digit: str,
):
    points = []
    segments = []
    with open(f'../../drawing/numbers/{digit}.txt', 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line:
                if line == '-1,-1':
                    points.append(segments[::5] + [segments[-1]])
                    segments = []
                else:
                    tmp = line.split(',')
                    segments.append((int(tmp[0]), int(tmp[1])))
    if segments:
        points.append(segments[::5] + [segments[-1]])

    drawer.points_groups = points
    drawer.invert_point_groups(rotation=1)
    drawer.resize_point_groups(keep_aspect_ratio=True, border=(1, 3))

    drawer.z = 72.57
    # drawer.cali_z(homing=False)
    # print(drawer.z)

    # if not drawer.pre_drawing(draw_bound=True):
    #     return

    for resized_points in drawer.points_groups:
        drawer.draw_points(resized_points)

    return None


if __name__ == '__main__':
    mt4 = MT4_UART()
    mt4.init(serial.Serial('/dev/tty.usbserial-21130', 115200), -1)
    mt4.zero()
    input()

    names = {0: '10C', 1: '10D', 2: '10H', 3: '10S', 4: '2C', 5: '2D', 6: '2H', 7: '2S', 8: '3C', 9: '3D', 10: '3H', 11: '3S', 12: '4C', 13: '4D', 14: '4H', 15: '4S', 16: '5C', 17: '5D', 18: '5H', 19: '5S', 20: '6C', 21: '6D', 22: '6H', 23: '6S', 24: '7C', 25: '7D', 26: '7H', 27: '7S', 28: '8C', 29: '8D', 30: '8H', 31: '8S', 32: '9C', 33: '9D', 34: '9H', 35: '9S', 36: 'AC', 37: 'AD', 38: 'AH', 39: 'AS', 40: 'JC', 41: 'JD', 42: 'JH', 43: 'JS', 44: 'KC', 45: 'KD', 46: 'KH', 47: 'KS', 48: 'QC', 49: 'QD', 50: 'QH', 51: 'QS'}
    model = YOLO("../yolov8s_playing_cards.pt")

    results = capture_and_detect(model)
    results = [names[i] for i in results]
    num_results = [1 if item[:-1] == 'A' else 10 if item[:-1] in ['J', 'Q', 'K'] else int(item[:-1]) for item in results]
    eq = solve24(num_results)

    print(eq)
    l = 20

    input()

    drawer = Drawing(robot=mt4)
    for i, letter in enumerate(eq):
        if letter == '/':
            letter = 'd'

        drawer.min_y = int(-120 + 240 / l * 2 * i)
        drawer.max_y = int(-120 + 240 / l * 2 * (i + 1))
        drawer.min_x = int(250 - (240 / 15))
        drawer.max_x = int(250 + (240 / 15))
        draw_digit(drawer, letter)
        drawer.wait_idle()