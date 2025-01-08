import cv2
import os
import time
import numpy as np

# Create a directory to save images
os.makedirs("images", exist_ok=True)

def capture_images(webcam_index=1, num_images=5, delay=2):
    """
    Captures overlapping images for panorama stitching.
    """
    cam = cv2.VideoCapture(webcam_index)
    if not cam.isOpened():
        print("Error: Could not access the webcam.")
        return []

    captured_images = []
    print("Capturing images...")
    for i in range(num_images):
        ret, frame = cam.read()
        if ret:
            enhanced_frame = enhance_brightness(frame)
            image_path = f"images/image_{i}.jpg"
            cv2.imwrite(image_path, enhanced_frame)
            captured_images.append(image_path)
            print(f"Captured and enhanced image {i + 1}")
        else:
            print(f"Failed to capture image {i + 1}")
        time.sleep(delay)  # Delay to allow rotation

    cam.release()
    return captured_images

def enhance_brightness(image):
    """
    Enhances the brightness of an image to improve visibility in low-light areas.
    """
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

def stitch_images(image_paths, gps_coords):
    """
    Stitches a series of images into a panorama and overlays GPS coordinates.
    """
    print("Stitching images...")
    images = [cv2.imread(img) for img in image_paths]
    stitcher = cv2.Stitcher_create()
    status, panorama = stitcher.stitch(images)

    if status == cv2.Stitcher_OK:
        print("Panorama created successfully.")
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
        
        # Overlay GPS coordinates on the panorama
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottom_left_corner = (10, new_height - 10)
        font_scale = 1
        font_color = (255, 255, 255)  # White
        line_type = 2
        
        cv2.putText(panorama_cropped, gps_coords, bottom_left_corner, font, font_scale, font_color, line_type)
        
        # Save the cropped panorama
        cv2.imwrite(panorama_path, panorama_cropped)
        print(f"Saved panorama as {panorama_path}")
        return panorama_path
    else:
        print("Error: Unable to stitch images.")
        return None

# Main process
if __name__ == "__main__":
    num_images = 8  # Adjust as needed
    delay_between_images = 2  # Adjust for rotation time
    gps_coords = "21.1671 N, 72.7852 E"  # Temporary variable for GPS coordinates

    # Step 1: Capture images
    image_paths = capture_images(num_images=num_images, delay=delay_between_images)
    if image_paths:
        panorama_path = stitch_images(image_paths, gps_coords)
        if panorama_path:
            print("Panorama completed.")
        else:
            print("Failed to create panorama.")
            