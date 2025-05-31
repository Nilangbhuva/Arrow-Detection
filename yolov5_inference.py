import sys
import cv2
import torch
from pathlib import Path

# Add the yolov5 directory to the system path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # assumes yolov5_inference.py is in yolov5 directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_coords
from utils.plots import Annotator

# Load the YOLOv5 model
model = DetectMultiBackend("/home/nilang/ros2_ws/yolov5/v5.pt", device='cpu')

# Initialize webcam (0 for default camera, change index for external cameras)
cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")

while True:
    # Read frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Convert frame to tensor
    img = torch.from_numpy(frame).to(model.device)
    img = img.permute(2, 0, 1).float()  # HWC to CHW
    img /= 255.0  # normalize to [0, 1]
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Perform object detection
    pred = model(img)

    # Apply NMS (Non-Maximum Suppression)
    pred = non_max_suppression(pred, 0.5, 0.45, classes=None, agnostic=False)

    # Process the results
    for det in pred:
        im0 = frame.copy()
        annotator = Annotator(im0, line_width=2, example=str(model.names))
        
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

            # Iterate over detections and draw bounding boxes
            for *xyxy, conf, cls in reversed(det):
                label = f'{model.names[int(cls)]} {conf:.2f}'
                annotator.box_label(xyxy, label, color=(255, 0, 0))
        
        # Display the annotated frame
        cv2.imshow("YOLOv5 Detection", annotator.result())

    # Press 'q' to quit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()