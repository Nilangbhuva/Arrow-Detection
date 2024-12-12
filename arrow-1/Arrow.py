import cv2
import numpy as np


cap=cv2.VideoCapture(0)
arrow1=cv2.imread("/home/yagnarao/Code/Rover/Right1.jpg",cv2.IMREAD_GRAYSCALE)
arrow2=cv2.imread("/home/yagnarao/Code/Rover/Right2.jpg",0)


while(True):
    _,frame=cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    # th=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    result1=cv2.matchTemplate(gray,arrow1,cv2.TM_CCOEFF_NORMED)
    result2=cv2.matchTemplate(gray,arrow2,cv2.TM_CCOEFF_NORMED)    

    w1=arrow1.shape[1]
    h1=arrow1.shape[0]
    w2=arrow2.shape[1]
    h2=arrow2.shape[0]
    threshold=0.7
    
    rectangles=[]
    xloc,yloc=np.where(result1>=threshold)
    for(x,y) in zip(xloc,yloc):
        rectangles.append([int(x),int(y),int(w1),int(h1)])
        rectangles.append([int(x),int(y),int(w1),int(h1)])

    rectangles,weights=cv2.groupRectangles(rectangles,1,0.2)
    for(x,y,w,h) in rectangles:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    rectangles=[]
    xloc,yloc=np.where(result2>=threshold)
    for(x,y) in zip(xloc,yloc):
        rectangles.append([int(x),int(y),int(w2),int(h2)])
        rectangles.append([int(x),int(y),int(w2),int(h2)])

    rectangles,weights=cv2.groupRectangles(rectangles,1,0.2)
    for(x,y,w,h) in rectangles:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)


    
    cv2.imshow('frame',gray)
    cv2.imshow('image1',arrow1)
    cv2.imshow('image2',arrow2)

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()