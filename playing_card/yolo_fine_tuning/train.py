from ultralytics import YOLO, checks, hub
checks()

hub.login('b7d75e16c36c81f0a65b2e5a29831c49a8cfdc99cc')

model = YOLO('https://hub.ultralytics.com/models/rm27ukQ5mHCQ5QfNWmjp')
results = model.train()