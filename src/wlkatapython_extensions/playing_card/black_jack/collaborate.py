import time
from typing import Tuple, List
import sys

sys.path.append("..")

import wlkatapython
import serial
from wlkatapython import MT4_UART, Mirobot_UART

import drawing

import cv2
import torch
import numpy as np
from collections import Counter
from ultralytics import YOLO
from PIL import Image

from ..model import solve24


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

    if len(results[0].boxes) != 0:
        box = results[0].boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
        label = f'{int(box.cls.item())}: {box.conf.item():.2f}'  # Class and confidence
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imwrite("detected_image.jpg", frame)

    return results


def flip_cards(
        mt4: MT4_UART,
        mirobot: Mirobot_UART,
        model: YOLO = None,
        names: dict[int, str] = None,
        card_deck_pos: Tuple[float, float, float] = (230, 129.9, 55.5),
        mt4_flip_pos: Tuple[float, float, float] = (230, 0, 209.9),
        mirobot_flip_pos: Tuple[float, float, float] = (260, 0, 162.7),
        mt4_move_away_pos: Tuple[float, float, float] = (199.2, 129.8, 269.9),
        dest_pos: Tuple[float, float, float] = (210, -60, 45.5),
        num_cards: int = 1,
        dest_offset: Tuple[float, float, float] = (0, 20, 0),
) -> List[str]:
    results = []

    for i in range(num_cards):
        card_deck_x, card_deck_y, card_deck_z = card_deck_pos
        card_deck_z -= i // 10

        mt4_flip_pos_x, mt4_flip_pos_y, mt4_flip_pos_z = mt4_flip_pos

        mirobot_flip_pos_x, mirobot_flip_pos_y, mirobot_flip_pos_z = mirobot_flip_pos

        # dest_pos_x = dest_pos[0] + i * dest_offset[0]
        # dest_pos_y = dest_pos[1] + i * dest_offset[1]
        # dest_pos_z = dest_pos[2] + i * dest_offset[2]

        dest_pos_x, dest_pos_y, dest_pos_z = dest_pos


        mt4.writecoordinate(0, 0, card_deck_x, card_deck_y, card_deck_z + 200, 0)
        mt4.writecoordinate(0, 0, card_deck_x, card_deck_y, card_deck_z + 30, 0)
        mt4.writecoordinate(0, 0, *card_deck_pos, 0)
        mt4.pump(1)
        while mt4.getState() != 'Idle':
            time.sleep(0.5)

        mt4.writecoordinate(0, 0, card_deck_x, card_deck_y, card_deck_z + 4, 0)
        mt4.writecoordinate(0, 0, card_deck_x, card_deck_y, card_deck_z + 2, 0)
        mt4.writecoordinate(0, 0, card_deck_x, card_deck_y, card_deck_z + 4, 0)
        while mt4.getState() != 'Idle':
            time.sleep(0.5)

        mt4.writecoordinate(0, 0, card_deck_x, card_deck_y, card_deck_z + 30, 0)
        mt4.writecoordinate(0, 0, card_deck_x, card_deck_y, card_deck_z + 200, 0)
        mt4.writecoordinate(0, 0, mt4_flip_pos_x, mt4_flip_pos_y, mt4_flip_pos_z + 60, 0)
        mt4.writecoordinate(0, 0, *mt4_flip_pos, 0)
        while mt4.getState() != 'Idle' or abs(float(mt4.getcoordinate(3)) - 209.9) > 1:
            time.sleep(0.5)

        mirobot.writecoordinate(0, 0, mirobot_flip_pos_x - 60, mirobot_flip_pos_y, mirobot_flip_pos_z, -81.9, -164.7, 87.8)
        mirobot.gripper(1)
        while mirobot.getState() != 'Idle':
            time.sleep(0.5)

        mirobot.writecoordinate(0, 0, *mirobot_flip_pos, -81.9, -164.7, 87.8)
        mirobot.gripper(2)
        while mirobot.getState() != 'Idle':
            time.sleep(0.5)

        mt4.pump(0)
        mt4.writecoordinate(0, 0, mt4_flip_pos_x, mt4_flip_pos_y, mt4_flip_pos_z + 60, 0)
        mt4.writecoordinate(0, 0, *mt4_move_away_pos, 0)
        while mt4.getState() != 'Idle':
            time.sleep(0.5)

        mirobot.writecoordinate(0, 0, *mirobot_flip_pos, -98, 15.3, 87.8)

        while mirobot.getState() != 'Idle':
            time.sleep(0.5)

        # Detection
        if model is not None:
            assert names is not None
            mt4.writecoordinate(0, 0, mt4_flip_pos_x + 15, mt4_flip_pos_y, mt4_flip_pos_z + 75, 0)
            mirobot.writecoordinate(0, 0, mirobot_flip_pos_x, mirobot_flip_pos_y - 20, mirobot_flip_pos_z - 20, -98, 15.3, 87.8)
            while mt4.getState() != 'Idle' or mirobot.getState() != 'Idle':
                time.sleep(0.5)
            result = capture_and_detect(model)
            try:
                card_name = names[int(result[0].boxes.cls[0].item())]
                print(result[0].boxes.cls.shape, card_name)
                results.append(card_name)
                if card_name[-1] == 'H':
                    dest_pos_y += 70
                elif card_name[-1] == 'C':
                    dest_pos_y -= 140
                elif card_name[-1] == 'D':
                    dest_pos_y -= 70
            except IndexError:
                dest_pos_x += 90
            mirobot.writecoordinate(0, 0, *mirobot_flip_pos, -98, 15.3, 87.8)

            while mirobot.getState() != 'Idle':
                time.sleep(0.5)

        mt4.writecoordinate(0, 0, mt4_flip_pos_x - 60, mt4_flip_pos_y, mt4_flip_pos_z + 60, 0)
        mt4.writecoordinate(0, 0, mt4_flip_pos_x - 60, mt4_flip_pos_y, mt4_flip_pos_z, 0)
        mt4.pump(1)

        while mt4.getState() != 'Idle':
            time.sleep(0.5)

        mirobot.writecoordinate(0, 0, *mirobot_flip_pos, -103, 10.3, 87.8)
        mirobot.gripper(1)
        mirobot.writecoordinate(0, 0, mirobot_flip_pos_x - 60, mirobot_flip_pos_y, mirobot_flip_pos_z, -98, 15.3, 87.8)
        mirobot.gripper(2)

        while mirobot.getState() != 'Idle':
            time.sleep(0.5)

        mirobot.writecoordinate(0, 0, 0, -198.7, 230.4, 0, 0, -90)
        mt4.writecoordinate(0, 0, dest_pos_x, dest_pos_y, dest_pos_z + 200, 0)
        mt4.writecoordinate(0, 0, dest_pos_x, dest_pos_y, dest_pos_z, 0)
        mt4.pump(0)
        mt4.zero()

        while mt4.getState() != 'Idle':
            time.sleep(0.5)

    return results

if __name__ == '__main__':
    # result = drawing.Drawing.detect_ports()
    # breakpoint()
    mt4 = MT4_UART()
    mt4.init(serial.Serial('/dev/tty.usbserial-1140', 115200), -1)
    mirobot = Mirobot_UART()
    mirobot.init(serial.Serial('/dev/tty.usbserial-1130', 115200), -1)
    mt4.pump(0)
    mt4.zero()
    mirobot.gripper(1)
    mirobot.writecoordinate(0, 0, 0, -198.7, 230.4, 0, 0, -90)

    names = {0: '10C', 1: '10D', 2: '10H', 3: '10S', 4: '2C', 5: '2D', 6: '2H', 7: '2S', 8: '3C', 9: '3D', 10: '3H', 11: '3S', 12: '4C', 13: '4D', 14: '4H', 15: '4S', 16: '5C', 17: '5D', 18: '5H', 19: '5S', 20: '6C', 21: '6D', 22: '6H', 23: '6S', 24: '7C', 25: '7D', 26: '7H', 27: '7S', 28: '8C', 29: '8D', 30: '8H', 31: '8S', 32: '9C', 33: '9D', 34: '9H', 35: '9S', 36: 'AC', 37: 'AD', 38: 'AH', 39: 'AS', 40: 'JC', 41: 'JD', 42: 'JH', 43: 'JS', 44: 'KC', 45: 'KD', 46: 'KH', 47: 'KS', 48: 'QC', 49: 'QD', 50: 'QH', 51: 'QS'}
    model = YOLO("../yolov8s_playing_cards.pt")
    while mirobot.getState() != 'Idle':
        time.sleep(0.5)
    mirobot.gripper(2)
    input()
    results = flip_cards(mt4, mirobot, model=model, names=names, num_cards=10)
    # num_results = [1 if item[:-1] == 'A' else int(item[:-1]) for item in results]
    # eq = solve24(num_results)
    # print(eq)
    mt4.zero()