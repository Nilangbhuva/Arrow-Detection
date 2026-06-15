from ultralytics import YOLO
import cv2
import numpy as np

# Load the model
model = YOLO("C:/Users/Nilang/Desktop/Arrow-Detection/arrow-Yolo/v8n.pt")

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_height, frame_width = frame.shape[:2]
    frame_center_x = frame_width // 2

    results = model.predict(source=frame, conf=0.4, show=False)

    for result in results:
        for detection in result.boxes:
            x1, y1, x2, y2 = detection.xyxy[0].int().tolist()
            object_center_x = (x1 + x2) // 2
            deviation_x = object_center_x - frame_center_x

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (object_center_x, (y1 + y2) // 2), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"Deviation X: {deviation_x}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Extract ROI and convert to grayscale
            roi = frame[y1:y2, x1:x2]
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # Split ROI into left and right halves
            h, w = gray_roi.shape
            left_half = gray_roi[:, :w // 2]
            right_half = gray_roi[:, w // 2:]

            # Compute average intensity
            left_intensity = np.mean(left_half)
            right_intensity = np.mean(right_half)

            # Determine direction
            direction = "Left" if left_intensity > right_intensity else "Right"

            # Display direction
            cv2.putText(frame, f"Arrow Direction: {direction}", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
