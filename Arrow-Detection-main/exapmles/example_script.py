# examples/example_script.py
import cv2
import os
import sys

# Ensure the project root directory is added to sys.path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.arrow_recognition import process_frame, match_and_annotate

# Define paths to the arrow images
right_arrow_path = os.path.join(os.path.dirname(__file__), '..', 'Right_Arrow.png')
left_arrow_path = os.path.join(os.path.dirname(__file__), '..', 'Left_Arrow.png')

# Load arrow images
right_arrow = cv2.imread(right_arrow_path, cv2.IMREAD_GRAYSCALE)
left_arrow = cv2.imread(left_arrow_path, cv2.IMREAD_GRAYSCALE)

# Verify images are loaded
if right_arrow is None:
    print(f"Error: Unable to load image at {right_arrow_path}")
    sys.exit(1)
if left_arrow is None:
    print(f"Error: Unable to load image at {left_arrow_path}")
    sys.exit(1)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Frame capture failed, skipping this frame.")
            continue

        # Check if frame is empty before processing
        if frame is None or frame.size == 0:
            print("Warning: Empty frame detected, skipping.")
            continue

        # Process the frame and perform arrow matching and annotation
        processed_frame = process_frame(frame)
        if processed_frame is None:
            print("Warning: Processed frame is empty, skipping.")
            continue

        # Only proceed with matching if processed frame is not empty
        match_and_annotate(processed_frame, right_arrow, (0, 255, 0), 'Right')
        match_and_annotate(processed_frame, left_arrow, (255, 0, 0), 'Left')

        # Display the processed video feed
        cv2.imshow("Video Feed", processed_frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
