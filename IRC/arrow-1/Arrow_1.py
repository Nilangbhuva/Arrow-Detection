import cv2
import numpy as np


cap=cv2.VideoCapture(0)
arrow=cv2.imread("/home/yagnarao/Code/Rover/Right.jpg",0)


while(True):
    _,frame=cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(gray,(5,5),0)


    _,thrash=cv2.threshold(gray,150,255,cv2.THRESH_BINARY)
    contors,_=cv2.findContours(thrash,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(frame,contors,-1,(0,255,0),2)

    for contor in contors:
        approx=cv2.approxPolyDP(contor,0.02*cv2.arcLength(contor,True),True)
        # cv2.drawContours(frame,[approx],0,(0,0,0),2)
        # x=approx.ravel()[0]
        # y=approx.ravel()[1]

        if len(approx)==3:
            x,y,w,h=cv2.boundingRect(contor)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,0),2)

    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()