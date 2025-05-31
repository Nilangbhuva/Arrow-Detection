import unittest
import cv2
import numpy as np
import sys
import os

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.arrow_recognition import edge_detection  # Ensure this path is correct

class TestEdgeDetection(unittest.TestCase):
    def test_edge_detection(self):
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Define the path to the image relative to the script directory
        image_path = os.path.join(script_dir, '..', 'Right_Arrow.png')  # Adjust if necessary
        
        # Normalize the path
        image_path = os.path.normpath(image_path)
        
        # Debugging: Print the image path and check existence
        print(f"Attempting to load image from: {image_path}")
        print(f"Does the image exist? {os.path.exists(image_path)}")
        
        # Load the image in grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        self.assertIsNotNone(image, f"Failed to load {image_path} for testing.")
        
        # Perform edge detection
        edges = edge_detection(image)
        
        # Assertions to verify edge detection output
        self.assertIsNotNone(edges, "Edge detection returned None.")
        self.assertEqual(edges.shape, image.shape, "Edge output shape mismatch with input image.")
        self.assertEqual(edges.dtype, np.uint8, "Edge output should be a binary image.")

if __name__ == '__main__':
    unittest.main()
