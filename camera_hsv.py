import numpy as np
import cv2

cap = cv2.VideoCapture(1)


def mouseEventHandler(event,x,y,flags,param):
    global currentimage
    if event == cv2.EVENT_MOUSEMOVE:
        print currentimage[y,x]
        

cv2.namedWindow('frame')
cv2.setMouseCallback('frame',mouseEventHandler)

while(True):
    global currentimage
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    #ret,thresh = cv2.threshold(gray,127,255,0)

    #contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    #cv2.drawContours(gray, contours, -1, (0,255,0), 2)
    
    # Display the resulting frame
    currentimage = gray
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
