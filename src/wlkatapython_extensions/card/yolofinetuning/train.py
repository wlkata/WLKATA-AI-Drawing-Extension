import os

from ultralytics import YOLO, checks, hub
checks()

hub.login(os.getenv('ultralytics_KEY'))

model = YOLO(os.getenv('ultralytics_MODEL_LINK'))
results = model.train()