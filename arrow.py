import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("custom_arrow_model_final.keras")

def detect_arrow_shape(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    edges = cv2.Canny(blurred, 50, 150)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if area < 500: 
            continue
        
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        
        if len(approx) >= 7:
            return True
    
    return False

cap = cv2.VideoCapture(0)
arrow_detected = False

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    potential_arrow = detect_arrow_shape(frame)
    
    if potential_arrow:
        img = cv2.resize(frame, (64, 64))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Make a prediction
        predictions = model.predict(img)
        class_idx = np.argmax(predictions)
        confidence = predictions[0][class_idx]
        
        if confidence > 0.75:
            arrow_detected = True
            arrow_direction = "Left Arrow" if class_idx == 0 else "Right Arrow"
            message = f"{arrow_direction} Detected with {confidence:.2f}"
            
            cv2.imwrite("detected_arrow.jpg", frame)
        else:
            message = "Arrow-like shape detected, but not confirmed"
            arrow_detected = False
    else:
        message = "No Arrow Detected"
        arrow_detected = False
    
    cv2.putText(frame, message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Arrow Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()