import cv2
import numpy as np

cap=cv2.VideoCapture(0)
arrow=cv2.imread("/home/yagnarao/Code/Rover/Right1.jpg",cv2.IMREAD_GRAYSCALE)

def convert_to_binary(frame):
    # Read the image
    original_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve Otsu's thresholding
    blurred_image = cv2.GaussianBlur(original_image, (5, 5), 0)

    # Apply Otsu's thresholding
    _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Normalize intensity values to the range [0, 1]
    normalized_image = binary_image.astype(np.uint8)

    # binary_image = binary_image.astype(np.uint8)

    return normalized_image
    # return binary_image

while(True):
    _,frame=cap.read()
    result=convert_to_binary(frame)

    # result_uint8 = (result * 255).astype(np.uint8)

     # th=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    result=cv2.matchTemplate(result,arrow,cv2.TM_CCOEFF_NORMED)

    w1=arrow.shape[1]
    h1=arrow.shape[0]

    threshold=0.5
    
    rectangles=[]
    xloc,yloc=np.where(result>=threshold)
    for(x,y) in zip(xloc,yloc):
        rectangles.append([int(x),int(y),int(w1),int(h1)])
        rectangles.append([int(x),int(y),int(w1),int(h1)])

    rectangles,weights=cv2.groupRectangles(rectangles,1,0.2)
    for(x,y,w,h) in rectangles:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    
    cv2.imshow('frame',frame)
    # cv2.imshow('image1',arrow1)
    # cv2.imshow('image2',arrow2)



    cv2.imshow('result',result)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cap.release()
