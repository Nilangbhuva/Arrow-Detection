from ultralytics import YOLO

model = YOLO("v8n.pt")
model.predict(source=0 , conf = 0.4, show=True)