import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

class YOLODetector(Node):
    def __init__(self):
        super().__init__('yolo_detector')
        
        # Create a subscriber for the camera feed
        self.subscription = self.create_subscription(
            Image,
            '/image',  # Subscribe to the camera topic
            self.image_callback,
            10)
        
        # Initialize the CV bridge
        self.bridge = CvBridge()
        
        # Load the YOLOv8 model
        self.model = YOLO("arrow-Yolo/best.pt")
        
        self.get_logger().info("YOLO Detector initialized. Press 'q' to quit.")

    def image_callback(self, msg):
        # Convert ROS Image message to OpenCV format
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # Perform object detection
        results = self.model.predict(source=frame, conf=0.5)
        
        # Get the plotted frame with detections
        annotated_frame = results[0].plot()
        
        # Display the annotated frame
        cv2.imshow("YOLOv8 Detection", annotated_frame)
        
        # Check for 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    yolo_detector = YOLODetector()
    rclpy.spin(yolo_detector)
    yolo_detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
