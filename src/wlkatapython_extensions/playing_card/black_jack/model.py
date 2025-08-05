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
    # model = YOLO("yolov8m.pt")

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
#
# import itertools
# import re
#
# def simplify_by_eval(expr, target=24, tol=1e-6):
#     def eval_expr_safe(s):
#         try:
#             return eval(s)
#         except ZeroDivisionError:
#             return None
#         except Exception:
#             return None
#
#     def find_innermost_pair(s):
#         stack = []
#         for i, char in enumerate(s):
#             if char == '(':
#                 stack.append(i)
#             elif char == ')':
#                 if stack:
#                     start = stack.pop()
#                     return start, i  # return the first innermost pair
#         return None
#
#     original_val = eval_expr_safe(expr)
#     if original_val is None:
#         return expr
#
#     while True:
#         pair = find_innermost_pair(expr)
#         if not pair:
#             break
#         start, end = pair
#         without = expr[:start] + expr[start+1:end] + expr[end+1:]
#         val = eval_expr_safe(without)
#         if val is not None and abs(val - target) < tol:
#             expr = without  # remove parens and restart
#         else:
#             # can't remove this one — skip it by replacing with placeholder to avoid infinite loop
#             expr = expr[:start] + '[' + expr[start+1:end] + ']' + expr[end+1:]
#
#     # restore any skipped ones
#     expr = expr.replace('[', '(').replace(']', ')')
#     return expr
#
#
# def solve24(nums):
#     ops = ['+', '-', '*', '/']
#     for p in itertools.permutations(nums):
#         for op1 in ops:
#             for op2 in ops:
#                 for op3 in ops:
#                     try:
#                         exprs = [
#                             f'({p[0]}{op1}{p[1]}){op2}({p[2]}{op3}{p[3]})',
#                             f'(({p[0]}{op1}{p[1]}){op2}{p[2]}){op3}{p[3]}',
#                             f'({p[0]}{op1}({p[1]}{op2}{p[2]})){op3}{p[3]}',
#                             f'{p[0]}{op1}(({p[1]}{op2}{p[2]}){op3}{p[3]})',
#                             f'{p[0]}{op1}({p[1]}{op2}({p[2]}{op3}{p[3]}))'
#                         ]
#                         for expr in exprs:
#                             if abs(eval(expr) - 24) < 1e-6:
#                                 return simplify_by_eval(expr)
#                     except ZeroDivisionError:
#                         continue
#     return None

# print(solve24([4, 5, 6, 7]))