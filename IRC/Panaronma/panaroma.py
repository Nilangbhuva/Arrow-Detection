import rclpy
from rclpy.node import Node
import cv2
import os
import time
import numpy as np

class PanoramaNode(Node):
    def __init__(self):
        super().__init__('panorama_node')

        # Declare ROS parameters
        self.declare_parameter('webcam_index', 0)
        self.declare_parameter('num_images', 5)
        self.declare_parameter('delay_between_images', 2)
        self.declare_parameter('gps_coords', "21.1671 N, 72.7852 E")

        # Create a directory to save images
        os.makedirs("images", exist_ok=True)

        self.get_logger().info("Panorama Node has started.")
        self.capture_and_stitch()

    def capture_and_stitch(self):
        webcam_index = self.get_parameter('webcam_index').get_parameter_value().integer_value
        num_images = self.get_parameter('num_images').get_parameter_value().integer_value
        delay = self.get_parameter('delay_between_images').get_parameter_value().integer_value
        gps_coords = self.get_parameter('gps_coords').get_parameter_value().string_value

        image_paths = self.capture_images(webcam_index, num_images, delay)
        if image_paths:
            panorama_path = self.stitch_images(image_paths, gps_coords)
            if panorama_path:
                self.get_logger().info(f"Panorama created and saved to {panorama_path}")
            else:
                self.get_logger().error("Failed to create panorama.")

    def capture_images(self, webcam_index, num_images, delay):
        """
        Captures overlapping images for panorama stitching.
        """
        cam = cv2.VideoCapture(webcam_index)
        if not cam.isOpened():
            self.get_logger().error("Could not access the webcam.")
            return []

        captured_images = []
        self.get_logger().info("Capturing images...")
        for i in range(num_images):
            ret, frame = cam.read()
            if ret:
                enhanced_frame = self.enhance_brightness(frame)
                image_path = f"images/image_{i}.jpg"
                cv2.imwrite(image_path, enhanced_frame)
                captured_images.append(image_path)
                self.get_logger().info(f"Captured and enhanced image {i + 1}")
            else:
                self.get_logger().error(f"Failed to capture image {i + 1}")
            time.sleep(delay)  # Delay to allow rotation

        cam.release()
        return captured_images

    def enhance_brightness(self, image):
        """
        Enhances the brightness of an image to improve visibility in low-light areas.
        """
        yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        y_channel = yuv_image[:, :, 0]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        y_channel_enhanced = clahe.apply(y_channel)
        yuv_image[:, :, 0] = y_channel_enhanced
        enhanced_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)
        return enhanced_image

    def stitch_images(self, image_paths, gps_coords):
        """
        Stitches a series of images into a panorama and overlays GPS coordinates.
        """
        self.get_logger().info("Stitching images...")
        images = [cv2.imread(img) for img in image_paths]
        stitcher = cv2.Stitcher_create()
        status, panorama = stitcher.stitch(images)

        if status == cv2.Stitcher_OK:
            self.get_logger().info("Panorama created successfully.")
            panorama_path = "panorama.jpg"

            height, width = panorama.shape[:2]
            new_width = width
            new_height = int(new_width / 3)

            if new_height > height:
                new_height = height
                new_width = 3 * new_height

            start_x = (width - new_width) // 2
            start_y = (height - new_height) // 2
            panorama_cropped = panorama[start_y:start_y+new_height, start_x:start_x+new_width]

            font = cv2.FONT_HERSHEY_SIMPLEX
            bottom_left_corner = (10, new_height - 10)
            font_scale = 1
            font_color = (255, 255, 255)
            line_type = 2

            cv2.putText(panorama_cropped, gps_coords, bottom_left_corner, font, font_scale, font_color, line_type)
            cv2.imwrite(panorama_path, panorama_cropped)
            return panorama_path
        else:
            self.get_logger().error("Unable to stitch images.")
            return None

def main(args=None):
    rclpy.init(args=args)
    node = PanoramaNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Panorama Node.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
