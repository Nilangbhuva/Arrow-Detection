import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("arrow-Yolo\\best.pt")

# Initialize webcam (0 for default camera, change index for external cameras)
cap = cv2.VideoCapture(0)

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

    # Perform object detection without show=True
    results = model.predict(source=frame, conf=0.5)
    
    # Get the plotted frame with detections
    annotated_frame = results[0].plot()
    
    # Display the annotated frame
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    # Press 'q' to quit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
