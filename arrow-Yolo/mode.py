from ultralytics import YOLO
model = YOLO("C:/Users/Nilang/Desktop/Arrow-Detection/arrow-Yolo/v8n.pt")
model.predict(source=0, conf=0.4, show=True)