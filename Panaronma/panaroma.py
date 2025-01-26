import os
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError

class ImageStitchingNode(Node):
    def __init__(self):
        super().__init__('image_stitching_node')

        # Create a directory to save images
        os.makedirs("images", exist_ok=True)
        
        self.bridge = CvBridge()
        
        # Subscribe to the image topic
        self.image_sub = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            10)
        
        # Create a service to trigger image capture
        self.capture_service = self.create_service(
            Trigger,
            'capture_images',
            self.capture_images_callback)

        self.image_paths = []
        self.capture_images = False
        self.num_images = 5
        self.delay_between_images = 2
        self.gps_data = {
            "latitude": 37.7749,
            "longitude": -122.4194,
            "altitude": 15.7,  # Example altitude
            "accuracy": 3.5,   # Example accuracy
        }
        self.scale = "Scale: 1:3"
        self.directions = "Cardinal Directions: N | S | E | W"
        self.image_count = 0

        # Start capturing images immediately
        self.start_capture()

    def start_capture(self):
        self.image_paths = []
        self.capture_images = True
        self.image_count = 0
        self.get_logger().info('Started capturing images.')

    def image_callback(self, msg):
        if not self.capture_images:
            return

        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Enhance brightness
            enhanced_image = self.enhance_brightness(cv_image)
            
            # Save the image
            image_path = f"images/image_{self.image_count}.jpg"
            cv2.imwrite(image_path, enhanced_image)
            self.image_paths.append(image_path)
            self.get_logger().info(f"Captured and enhanced image {self.image_count + 1}")
            
            self.image_count += 1
            if self.image_count >= self.num_images:
                self.capture_images = False
                self.stitch_images(self.image_paths, self.gps_data, self.scale, self.directions)
        except CvBridgeError as e:
            self.get_logger().error(f"Failed to convert image: {e}")

    def capture_images_callback(self, request, response):
        self.start_capture()
        response.success = True
        response.message = 'Started capturing images.'
        return response

    def enhance_brightness(self, image):
        # Convert the image to YUV color space
        yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        
        # Extract the Y (luminance) channel
        y_channel = yuv_image[:, :, 0]
        
        # Apply CLAHE to the Y channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        y_channel_enhanced = clahe.apply(y_channel)
        
        # Merge the enhanced Y channel back with the U and V channels
        yuv_image[:, :, 0] = y_channel_enhanced
        enhanced_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)
        
        return enhanced_image

    def stitch_images(self, image_paths, gps_data, scale, directions):
        self.get_logger().info("Stitching images...")
        images = [cv2.imread(img) for img in image_paths]
        stitcher = cv2.Stitcher_create()
        status, panorama = stitcher.stitch(images)

        if status == cv2.Stitcher_OK:
            self.get_logger().info("Panorama created successfully.")
            panorama_path = "panorama.jpg"
            
            # Calculate the new dimensions to maintain an aspect ratio of 1:3
            height, width = panorama.shape[:2]
            new_width = width
            new_height = int(new_width / 3)
            
            # Ensure the new height is not greater than the current height
            if new_height > height:
                new_height = height
                new_width = 3 * new_height
            
            # Crop the panorama to the new dimensions
            start_x = (width - new_width) // 2
            start_y = (height - new_height) // 2
            panorama_cropped = panorama[start_y:start_y+new_height, start_x:start_x+new_width]
            
            # Overlay GPS coordinates, scale, and directions on the panorama
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_color = (255, 255, 255)  # White
            line_type = 2
            
            gps_text = f"Latitude: {gps_data['latitude']}, Longitude: {gps_data['longitude']}, Altitude: {gps_data['altitude']}m, Accuracy: {gps_data['accuracy']}m"
            cv2.putText(panorama_cropped, gps_text, (10, new_height - 60), font, font_scale, font_color, line_type)
            cv2.putText(panorama_cropped, scale, (10, new_height - 30), font, font_scale, font_color, line_type)
            cv2.putText(panorama_cropped, directions, (10, new_height - 10), font, font_scale, font_color, line_type)
            
            # Save the cropped panorama
            cv2.imwrite(panorama_path, panorama_cropped)
            self.get_logger().info(f"Saved panorama as {panorama_path}")
        else:
            self.get_logger().error("Error: Unable to stitch images.")

def main(args=None):
    rclpy.init(args=args)
    image_stitching_node = ImageStitchingNode()
    rclpy.spin(image_stitching_node)
    image_stitching_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()