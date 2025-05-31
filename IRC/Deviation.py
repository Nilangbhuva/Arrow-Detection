import cv2
import numpy as np

def detect_arrow(frame):
    """
    Detect the arrow shape in the frame.
    Assumes the arrow is black on a white background.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding to isolate black shapes on white
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Filter out small objects by area
        area = cv2.contourArea(contour)
        if area < 1000:  # Ignore very small objects
            continue

        # Calculate bounding box and aspect ratio
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h

        # Check if the shape resembles an arrow
        if 0.5 < aspect_ratio < 2.0:  # Aspect ratio typical of arrows
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

            # Check if contour has enough vertices to resemble an arrow
            if len(approx) >= 7:
                return contour, binary, (x, y, w, h)  # Return arrow contour, binary image, and bounding box

    return None, binary, None  # Return None if no arrow is detected

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get frame dimensions
    frame_height, frame_width, _ = frame.shape
    frame_center_x = frame_width // 2

    # Detect arrow in the frame
    arrow_contour, binary_frame, bounding_box = detect_arrow(frame)

    if arrow_contour is not None:
        # Highlight the detected arrow contour
        cv2.drawContours(frame, [arrow_contour], -1, (0, 255, 0), 2)

        # Calculate the center of the arrow
        x, y, w, h = bounding_box
        arrow_center_x = x + w // 2

        # Draw the center point of the arrow
        cv2.circle(frame, (arrow_center_x, y + h // 2), 5, (0, 0, 255), -1)

        # Calculate deviation from the center of the frame
        deviation_x = arrow_center_x - frame_center_x

        # Determine the direction
        direction = "Left" if deviation_x < 0 else "Right"

        # Display deviation and direction information
        deviation_message = f"Deviation: {abs(deviation_x)}px ({direction})"
        cv2.putText(frame, deviation_message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Draw a line from the frame center to the arrow center
        cv2.line(frame, (frame_center_x, y + h // 2), (arrow_center_x, y + h // 2), (255, 0, 0), 2)
    else:
        cv2.putText(frame, "No Arrow Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Display the live feeds
    cv2.imshow("Normal Feed", frame)  # Normal feed with deviation and arrow info
    cv2.imshow("Binary Feed", binary_frame)  # Binary feed highlighting black shapes

    # Exit the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
