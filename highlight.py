import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained Keras model
model = load_model("custom_arrow_model_final.keras")

def detect_arrow_shape(frame):
    """
    Detect the arrow-like shape in the given frame.
    Assumes the arrow is black on a white background.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding to isolate black shapes on white
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Calculate the area and perimeter
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        # Skip small contours
        if area < 500:
            continue

        # Approximate the contour shape
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        # Check for arrow-like shape (7+ vertices)
        if len(approx) >= 7:
            return contour, binary  # Return the arrow contour and binary image

    return None, binary  # Return None and the binary image if no arrow is found

# Capture video feed from the default camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect arrow in the frame
    arrow_contour, binary_frame = detect_arrow_shape(frame)

    if arrow_contour is not None:
        # Highlight the detected arrow contour
        cv2.drawContours(frame, [arrow_contour], -1, (0, 255, 0), 2)

        # Extract the bounding box of the arrow
        x, y, w, h = cv2.boundingRect(arrow_contour)
        cropped_arrow = frame[y:y+h, x:x+w]

        # Resize and preprocess the cropped arrow for model input
        img = cv2.resize(cropped_arrow, (64, 64))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB if needed
        img = img / 255.0  # Normalize pixel values
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        # Use the model to predict the arrow direction
        predictions = model.predict(img)
        class_idx = np.argmax(predictions)
        confidence = predictions[0][class_idx]

        if confidence > 0.75:
            arrow_direction = "Left Arrow" if class_idx == 0 else "Right Arrow"
            message = f"{arrow_direction} Detected with {confidence:.2f}"
            cv2.imwrite("detected_arrow.jpg", frame)
        else:
            message = "Arrow-like shape detected, but not confirmed"
    else:
        message = "No Arrow Detected"

    # Display the message on the frame
    cv2.putText(frame, message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the live feeds
    cv2.imshow("Normal Feed", frame)       # Normal feed with arrow contours and message
    cv2.imshow("Grayscale Feed", binary_frame)  # Binary feed highlighting black shapes

    # Exit the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
