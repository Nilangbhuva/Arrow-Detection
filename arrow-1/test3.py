import cv2

cap = cv2.VideoCapture(0)

def convert_to_binary(original_image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve Otsu's thresholding
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Apply Otsu's thresholding
    _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Normalize intensity values to the range [0, 1]
    # normalized_image = binary_image / 255.0

    return binary_image


while True:
    _, frame = cap.read()
    result = convert_to_binary(frame)
    
   
    cv2.imshow('result', result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
