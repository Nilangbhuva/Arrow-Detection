cd ~/iot-smart-home

cat > arrow_detection.py <<'PY'
import cv2
import time
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO("v8n.pt")

# Open USB webcam
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

cap.set(cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

print("Arrow Detection Started!")
print("Press Q to quit")

# Black pixel threshold
BLACK_THRESHOLD = 80

# Minimum difference required to classify direction
DIRECTION_THRESHOLD = 1.15

prev_time = time.time()

while True:

    ret, frame = cap.read()

    if not ret:
        print("Frame capture failed")
        break

    # YOLO inference
    results = model.predict(
        frame,
        imgsz=416,
        conf=0.35,
        device="cpu",
        verbose=False
    )

    output = frame.copy()

    for result in results:

        boxes = result.boxes

        if boxes is None:
            continue

        for box in boxes:

            # Get bounding box coordinates
            x1, y1, x2, y2 = map(
                int, box.xyxy[0].tolist()
            )

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            if x2 <= x1 or y2 <= y1:
                continue

            # Crop detected arrow
            roi = frame[y1:y2, x1:x2]

            # Convert to grayscale
            gray = cv2.cvtColor(
                roi, cv2.COLOR_BGR2GRAY
            )

            # Detect black pixels
            _, binary = cv2.threshold(
                gray,
                BLACK_THRESHOLD,
                255,
                cv2.THRESH_BINARY_INV
            )

            # Divide bounding box into two halves
            mid = binary.shape[1] // 2

            left_half = binary[:, :mid]
            right_half = binary[:, mid:]

            # Count black pixels
            left_black = cv2.countNonZero(left_half)
            right_black = cv2.countNonZero(right_half)

            # Determine direction
            if left_black > right_black * DIRECTION_THRESHOLD:
                direction = "LEFT"
                color = (0, 255, 0)

            elif right_black > left_black * DIRECTION_THRESHOLD:
                direction = "RIGHT"
                color = (255, 0, 0)

            else:
                direction = "UNCERTAIN"
                color = (0, 255, 255)

            # Confidence from YOLO
            confidence = float(box.conf[0])

            # Draw bounding box
            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            # Draw dividing line
            cv2.line(
                output,
                (x1 + mid, y1),
                (x1 + mid, y2),
                (0, 255, 255),
                2
            )

            # Display direction and confidence
            label = f"{direction} | {confidence:.2f}"

            cv2.putText(
                output,
                label,
                (x1, max(y1 - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2
            )

            # Display black pixel counts
            cv2.putText(
                output,
                f"L:{left_black} R:{right_black}",
                (x1, min(y2 + 22, output.shape[0] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )

    # FPS calculation
    current_time = time.time()
    fps = 1 / max(current_time - prev_time, 0.001)
    prev_time = current_time

    cv2.putText(
        output,
        f"FPS: {fps:.1f}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("YOLO Arrow Direction Detection", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Detection stopped")
PY
