import cv2
import numpy as np
import math

cap = cv2.VideoCapture(0)
right = cv2.imread("original.jpg", cv2.IMREAD_GRAYSCALE)
threshold=0.8

def preprocess(img):
    img_canny = cv2.Canny(img, 50, 50)
    kernel = np.ones((3, 3))
    img_dilate = cv2.dilate(img_canny, kernel, iterations=2)
    img_erode = cv2.erode(img_dilate, kernel, iterations=1)
    return img_erode

def find_tip(points, convex_hull):
    length = len(points)
    indices = np.setdiff1d(range(length), convex_hull)

    for i in range(2):
        j = indices[i] + 2
        if j > length - 1:
            j = length - j
        if np.all(points[j] == points[indices[i - 1] - 2]):
            return tuple(points[j])
        
def convert_to_binary(frame):
    original_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(original_image, (5, 5), 0)
    _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # normalized_image = binary_image.astype(np.uint8)
    return binary_image


def direction(approx):
    l=r=0
    for a in approx:
        if(a[0,0]>arrow_tip[0]):
            l+=1
        if(a[0,0]<arrow_tip[0]):
            r+=1

    if(l>4):
        return -1
    if(r>4):
        return 1



def mean(approx):
    sumx=sumy=0
    for a in approx:
        sumx+=a[0,0]
        sumy+=a[0,1]
    centroid=[]
    centroid.append((sumx/7)-320)
    centroid.append((sumy/7)-240)
    return centroid

while True:
    _, frame = cap.read()
    original_height, original_width = frame.shape[:2]

    # Set desired width or height while keeping the aspect ratio
    width = 640  # Target width (for example)
    aspect_ratio = original_width / original_height
    height = int(width / aspect_ratio)
    frame = cv2.resize(frame, (width, height))

    # Calculate the center and deviation range
    center_x = width // 2
    tolerance = int(0.1 * width)  # ±10% of the frame width

    min_x = center_x - tolerance  # Left bound of tolerance
    max_x = center_x + tolerance  # Right bound of tolerance

    frame1 = convert_to_binary(frame)
    contours, hierarchy = cv2.findContours(preprocess(frame1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.01 * peri, True)
        hull = cv2.convexHull(approx, returnPoints=False)
        sides = len(hull)

        if 6 > sides > 3 and sides + 2 == len(approx):
            arrow_tip = find_tip(approx[:, 0, :], hull.squeeze())
            if arrow_tip:
                cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 3)
                cv2.circle(frame, arrow_tip, 3, (0, 0, 255), cv2.FILLED)

                # Calculate centroid
                centroid = mean(approx)
                arrow_centroid_x = int(centroid[0] + center_x)  # Adjusting with center_x

                # Draw the centroid on the frame
                cv2.circle(frame, (arrow_centroid_x, center_x), 5, (255, 0, 0), cv2.FILLED)

                # Calculate x-axis deviation
                deviation_x = arrow_centroid_x - center_x

                # Determine arrow direction
                arrow_direction = direction(approx)
                direction_text = "Right" if arrow_direction == 1 else "Left"

                # Determine if deviation is acceptable
                if min_x <= arrow_centroid_x <= max_x:
                    deviation_status = "Within Tolerance"
                    color = (0, 255, 0)  # Green
                else:
                    deviation_status = "Out of Tolerance"
                    color = (0, 0, 255)  # Red

                # Display deviation and direction on the frame
                cv2.putText(frame, f"Deviation X: {deviation_x} ({deviation_status})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, f"Arrow Direction: {direction_text}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                print(f"Deviation X: {deviation_x} ({deviation_status}), Direction: {direction_text}")

    # Draw the frame center line (thicker for better visibility)
    cv2.line(frame, (center_x, 0), (center_x, height), (255, 0, 0), 3)

    # Draw tolerance range lines
    cv2.line(frame, (min_x, 0), (min_x, height), (0, 255, 0), 2)
    cv2.line(frame, (max_x, 0), (max_x, height), (0, 255, 0), 2)

    cv2.imshow("Image", frame)
    cv2.imshow("Image1", frame1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
