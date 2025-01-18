from ultralytics import YOLO
import cv2

# Load the model
model = YOLO("v8n.pt")

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Get frame dimensions
    frame_height, frame_width = frame.shape[:2]
    frame_center_x = frame_width // 2

    # Predict using the model
    results = model.predict(source=frame, conf=0.4, show=False)

    for result in results:
        # Iterate over each detection
        for detection in result.boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = detection.xyxy[0].int().tolist()
            # Calculate the center of the detected object
            object_center_x = (x1 + x2) // 2
            
            # Calculate the deviation in the X direction
            deviation_x = object_center_x - frame_center_x
            
            # Draw the bounding box and center point on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (object_center_x, (y1 + y2) // 2), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"Deviation X: {deviation_x}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Display the frame with the bounding box and deviation
    cv2.imshow("Frame", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()