from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Train the model
model.train(data='data.yaml', epochs=100, imgsz=640)