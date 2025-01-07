import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
import cv2
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO
from geometry_msgs.msg import Twist

class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_node')
        self.publisher_ = self.create_publisher(Twist, 'yolo_info', 10)
        self.subscription = self.create_subscription(
            Image,
            'image_raw',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

        self.bridge = CvBridge()
        self.model = YOLO("/home/nilang/ros2_ws/src/py_pubsub/py_pubsub/best.pt")
        self.get_logger().info('YOLO node has been started.')

    def listener_callback(self, data):
        try:
            frame = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')
            return

        # Perform object detection
        results = self.model.predict(source=frame, conf=0.5)

        frame_height, frame_width = frame.shape[:2]
        frame_center_x = frame_width // 2

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bbox_center_x = (x1 + x2) // 2
                deviation_x = bbox_center_x - frame_center_x

                # Update message publishing to use Twist
                msg = Twist()
                msg.linear.y = float(deviation_x)  # Store deviation in linear.y like arrow code
                msg.linear.z = 1.0  # Indicate detection (similar to direction in arrow code)
                self.publisher_.publish(msg)
                self.get_logger().info(f'Published: deviation={deviation_x}')

                # Draw visualization
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.circle(frame, (bbox_center_x, (y1 + y2) // 2), 5, (0, 0, 255), -1)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Detection with Deviation", frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    yolo_node = YoloNode()
    rclpy.spin(yolo_node)
    yolo_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()