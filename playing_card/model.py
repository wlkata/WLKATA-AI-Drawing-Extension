import cv2
import torch
import numpy as np
from collections import Counter
from ultralytics import YOLO
from PIL import Image
import time


def get_most_frequent_cards(detections):
    """
    Process the detections to determine the most frequently occurring card types
    and eliminate duplicate detections of the same card.
    """
    all_detected_cards = []

    for detection in detections:
        for result in detection:
            card_set = set()
            if result.boxes is not None:
                for box in result.boxes:
                    cls = int(box.cls.item())
                    card_set.add(cls)
            all_detected_cards.append(frozenset(card_set))
    breakpoint()
    # Count occurrences of each card class
    card_counts = Counter(all_detected_cards)

    # Get unique detected cards
    unique_cards = list(card_counts.keys())

    return unique_cards


def main():
    # Load YOLO model
    model = YOLO("yolov8s_playing_cards.pt")  # Replace with the actual path to your model

    if True:
        model(0, show=True, device='mps', vid_stride=1, show_conf=False)
        # from pathlib import Path
        # import random
        # import shutil
        #
        # images_folder = Path('yolo_fine_tuning/poker/images')
        # names = []
        # for file in images_folder.iterdir():
        #     if file.suffix == '.jpg':
        #         names.append(file.stem)
        #
        # random.shuffle(names)
        # train_ratio = 0.8
        # val_ratio = 0.1
        #
        # train_size = int(len(names) * train_ratio)
        # val_size = int(len(names) * val_ratio)
        #
        # train_list = names[:train_size]
        # val_list = names[train_size:train_size + val_size]
        # test_list = names[train_size + val_size:]
        #
        # for name in train_list:
        #     shutil.move(f'yolo_fine_tuning/poker/images/{name}.jpg', f'yolo_fine_tuning/poker/images/train/{name}.jpg')
        #     shutil.move(f'yolo_fine_tuning/poker/labels/{name}.txt', f'yolo_fine_tuning/poker/labels/train/{name}.txt')
        #
        # for name in val_list:
        #     shutil.move(f'yolo_fine_tuning/poker/images/{name}.jpg', f'yolo_fine_tuning/poker/images/val/{name}.jpg')
        #     shutil.move(f'yolo_fine_tuning/poker/labels/{name}.txt', f'yolo_fine_tuning/poker/labels/val/{name}.txt')
        #
        # for name in test_list:
        #     shutil.move(f'yolo_fine_tuning/poker/images/{name}.jpg', f'yolo_fine_tuning/poker/images/test/{name}.jpg')
        #     shutil.move(f'yolo_fine_tuning/poker/labels/{name}.txt', f'yolo_fine_tuning/poker/labels/test/{name}.txt')
        #
        # print(len(train_list), len(val_list), len(test_list))

    else:
        # Open webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        detections = []
        input("Press Enter to start...")
        start_time = time.time()

        while time.time() - start_time < 5:  # Run for 5 seconds
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to RGB for YOLO compatibility
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert to PIL Image
            img_pil = Image.fromarray(frame_rgb)

            # Run YOLO prediction
            results = model(img_pil, device='mps')
            detections.append(results)

            # Show frame (Optional for debugging)
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Process detections to find the most frequent cards
        most_frequent_cards = get_most_frequent_cards(detections)

        print("Detected Cards:", most_frequent_cards)

        # Release resources
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

import itertools

def solve24(nums):
    ops = ['+', '-', '*', '/']
    for p in itertools.permutations(nums):
        for op1 in ops:
            for op2 in ops:
                for op3 in ops:
                    try:
                        # (a op b) op (c op d)
                        if abs(eval(f'({p[0]}{op1}{p[1]}){op2}({p[2]}{op3}{p[3]})') - 24) < 1e-6:
                            return f'({p[0]}{op1}{p[1]}){op2}({p[2]}{op3}{p[3]})'
                        # ((a op b) op c) op d
                        if abs(eval(f'(({p[0]}{op1}{p[1]}){op2}{p[2]}){op3}{p[3]}') - 24) < 1e-6:
                            return f'(({p[0]}{op1}{p[1]}){op2}{p[2]}){op3}{p[3]}'
                        # (a op (b op c)) op d
                        if abs(eval(f'({p[0]}{op1}({p[1]}{op2}{p[2]})){op3}{p[3]}') - 24) < 1e-6:
                            return f'({p[0]}{op1}({p[1]}{op2}{p[2]})){op3}{p[3]}'
                        # a op ((b op c) op d)
                        if abs(eval(f'{p[0]}{op1}(({p[1]}{op2}{p[2]}){op3}{p[3]})') - 24) < 1e-6:
                            return f'{p[0]}{op1}(({p[1]}{op2}{p[2]}){op3}{p[3]})'
                        # a op (b op (c op d))
                        if abs(eval(f'{p[0]}{op1}({p[1]}{op2}({p[2]}{op3}{p[3]}))') - 24) < 1e-6:
                            return f'{p[0]}{op1}({p[1]}{op2}({p[2]}{op3}{p[3]}))'
                    except ZeroDivisionError:
                        continue
    return None
#
# print(solve24([1, 3, 9, 10]))