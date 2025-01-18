import cv2
from ultralytics import YOLO

class YoloDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        print('YOLO detector has been initialized.')

    def process_frame(self, frame):
        # Convert the frame to grayscale to reduce the computational load
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Convert the grayscale image back to a 3-channel image
        gray_frame_3_channel = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

        # Perform object detection
        results = self.model.predict(source=gray_frame_3_channel, conf=0.5)

        frame_height, frame_width = frame.shape[:2]
        frame_center_x = frame_width // 2

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bbox_center_x = (x1 + x2) // 2
                deviation_x = bbox_center_x - frame_center_x

                print(f'Deviation: {deviation_x}')

                # Draw visualization on the original frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.circle(frame, (bbox_center_x, (y1 + y2) // 2), 5, (0, 0, 255), -1)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Detection with Deviation", frame)
        cv2.waitKey(1)

def main():
    model_path = "arrow-Yolo\\best.pt"
    yolo_detector = YoloDetector(model_path)

    cap = cv2.VideoCapture(0)  # Use the first webcam
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        yolo_detector.process_frame(frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()