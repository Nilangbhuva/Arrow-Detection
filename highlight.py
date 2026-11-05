cd ~/iot-smart-home

cat > arrow_detection.py <<'PY'
import cv2
import time
from gpiozero import Servo

# Servo setup: GPIO18 (physical pin 12)
servo = Servo(
    18,
    min_pulse_width=0.0005,
    max_pulse_width=0.0025
)

LEFT_POSITION = -0.5
RIGHT_POSITION = 0.5

servo.mid()
time.sleep(1)

# USB webcam - use the same initialization as camera_test.py
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not detected!")
    servo.detach()
    exit()

# Request a standard resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

last_direction = None
last_move_time = 0
COOLDOWN = 1.0

def detect_arrow(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Dark arrow on light background
    _, thresh = cv2.threshold(
        gray, 100, 255, cv2.THRESH_BINARY_INV
    )

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < 1500:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        if w < 50 or h < 30 or w / h < 1.2:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(
            contour, 0.025 * perimeter, True
        )

        points = approx.reshape(-1, 2)

        center_y = y + h / 2

        left_point = points[points[:, 0].argmin()]
        right_point = points[points[:, 0].argmax()]

        if abs(left_point[1] - center_y) < h * 0.20:
            return "LEFT", (x, y, w, h)

        if abs(right_point[1] - center_y) < h * 0.20:
            return "RIGHT", (x, y, w, h)

    return None, None


print("Arrow detection running. Press Q to quit.")

try:
    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            print("Camera frame failed")
            time.sleep(0.1)
            continue

        direction, box = detect_arrow(frame)

        if direction:
            x, y, w, h = box

            cv2.rectangle(
                frame, (x, y), (x+w, y+h), (0, 255, 0), 2
            )

            cv2.putText(
                frame, direction + " ARROW",
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2
            )

            current_time = time.time()

            if (direction != last_direction and
                current_time - last_move_time > COOLDOWN):

                if direction == "LEFT":
                    print("LEFT detected - Servo moving LEFT")
                    servo.value = LEFT_POSITION

                else:
                    print("RIGHT detected - Servo moving RIGHT")
                    servo.value = RIGHT_POSITION

                last_direction = direction
                last_move_time = current_time

        cv2.imshow("Arrow Detection and Servo Control", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    servo.mid()
    time.sleep(1)
    servo.detach()
    print("Program stopped")


PY
