cat > arrow_detection.py <<'PY'
import cv2
import time
from gpiozero import Servo

# Servo connected to BCM GPIO18 (physical pin 12)
servo = Servo(
    18,
    min_pulse_width=0.0005,
    max_pulse_width=0.0025
)

# Servo positions
LEFT_POSITION = -0.5
RIGHT_POSITION = 0.5

# Start at centre
servo.mid()
time.sleep(1)

# USB camera connected to Raspberry Pi
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Camera not detected!")
    servo.detach()
    exit()

last_direction = None
last_move_time = 0
COOLDOWN = 1.0

def detect_arrow(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Threshold dark arrow printed on white paper
    _, thresh = cv2.threshold(
        gray, 100, 255, cv2.THRESH_BINARY_INV
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < 1500:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(
            contour, 0.025 * perimeter, True
        )

        x, y, w, h = cv2.boundingRect(contour)

        if w < 50 or h < 30:
            continue

        # Ignore shapes that are too narrow or too tall
        if w / h < 1.2:
            continue

        points = approx.reshape(-1, 2)
        center_y = y + h / 2

        # Find the leftmost and rightmost polygon vertices
        left_point = points[points[:, 0].argmin()]
        right_point = points[points[:, 0].argmax()]

        left_center_distance = abs(left_point[1] - center_y)
        right_center_distance = abs(right_point[1] - center_y)

        # The arrow tip is usually near the vertical centre
        if left_center_distance < h * 0.20:
            return "LEFT", (x, y, w, h)

        if right_center_distance < h * 0.20:
            return "RIGHT", (x, y, w, h)

    return None, None


print("Arrow detection started. Press Q to quit.")

try:
    while True:
        ret, frame = camera.read()

        if not ret:
            print("Camera frame not received")
            break

        direction, box = detect_arrow(frame)

        if direction:
            x, y, w, h = box

            cv2.rectangle(
                frame, (x, y), (x+w, y+h), (0, 255, 0), 2
            )

            cv2.putText(
                frame,
                direction + " ARROW",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            current_time = time.time()

            if (direction != last_direction and
                    current_time - last_move_time > COOLDOWN):

                if direction == "LEFT":
                    print("LEFT arrow detected - Servo LEFT")
                    servo.value = LEFT_POSITION

                elif direction == "RIGHT":
                    print("RIGHT arrow detected - Servo RIGHT")
                    servo.value = RIGHT_POSITION

                last_direction = direction
                last_move_time = current_time

        cv2.imshow("Raspberry Pi Arrow Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
    servo.mid()
    time.sleep(1)
    servo.detach()
    print("Program stopped safely.")
PY
