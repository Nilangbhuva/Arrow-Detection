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
    # Resize the frame
    frame = cv2.resize(frame, (width, height))
    frame1=convert_to_binary(frame)
    contours, hierarchy = cv2.findContours(preprocess(frame1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.01 * peri, True)
        hull = cv2.convexHull(approx, returnPoints=False)
        sides = len(hull)

        if 6 > sides > 3 and sides + 2 == len(approx):
            arrow_tip = find_tip(approx[:,0,:], hull.squeeze())
            if arrow_tip:
                cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 3)
                cv2.circle(frame, arrow_tip, 3, (0, 0, 255), cv2.FILLED)
                print(direction(approx))
                print(mean(approx))

    cv2.imshow("Image", frame)
    cv2.imshow("Image1", frame1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
