import cv2
import numpy as np
from numpy import int32
import math

def binarize(im):
    '''Turn into white any portion of the image that is not zero'''
    new = np.zeros_like(im, dtype=np.uint8)
    new[im > 1] = 255
    return new

def threshold_range(im, lo, hi):
    '''Returns a binary image if the values are between a certain value'''

    unused, t1 = cv2.threshold(im, lo, 255, type=cv2.THRESH_BINARY)
    unused, t2 = cv2.threshold(im, hi, 255, type=cv2.THRESH_BINARY_INV)
    return cv2.bitwise_and(t1, t2)

Y_IMAGE_RES = 760 #resolution i guess
VIEW_ANGLE  = 34.8665269 #unsure
TARGET_HEIGHT = 7 #inches ##############<<<<<<<<<<<<CHANGE THIS FOR ROBOT
PI = 3.14159265358979 #yeah, you know it
VISION_CONST = 4 #magic number. Adjust as necessary

def calculate_distance(height):
    global Y_IMAGE_RES
    global VIEW_ANGLE
    global TARGET_HEIGHT
    global PI
    global VISION_CONST

    return (Y_IMAGE_RES * TARGET_HEIGHT / (height * 12 * 2 * math.tan(VIEW_ANGLE * PI / (180*2))))**VISION_CONST

cap = cv2.VideoCapture(1)

orb = cv2.ORB()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

template = cv2.imread("vision_target_binary.png",0)

kp1, des1 = orb.detectAndCompute(template,None)

FRAME_WIDTH = cap.get(3)
FRAME_HEIGHT = cap.get(4)
SCREEN_MIDPOINT = FRAME_WIDTH/2

while(True):

    ret, img = cap.read()

    # convert to HSV colorspace
    hsv = cv2.cvtColor(img, cv2.cv.CV_BGR2HSV)

    h, s, v = cv2.split(hsv)

    # these parameters will find 'green' on the image
    h = threshold_range(h, 0, 100) ## h, 30, 75 original
    s = threshold_range(s, 0, 30) ## s, 188, 255 original
    v = threshold_range(v, 250, 255)

    # combine them all and show that
    combined = cv2.bitwise_and(h, cv2.bitwise_and(s, v))

    # fill in the holes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3), anchor=(1,1)) #original (cv2.MORPH_RECT, (2,2), anchor=(1,1))
    morphed_img = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=12) #original 9 iterations MORPH_CLOSE

    # nothing easier
    contours, hierarchy = cv2.findContours(morphed_img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS) #original CHAIN_APPROX_TC89_KCOS
    #print contours

    #loop through and remove if not convex
    for cont in contours:
        if cv2.isContourConvex(cont):
            try:
                contours.remove(cont)
            except ValueError:
                pass

    # draw the found contours on the image
    # -> but you can't show colors on a grayscale image, so convert it to color
    #color_img = cv2.cvtColor(morphed_img, cv2.cv.CV_GRAY2BGR)

    length = 0
    heights = []
    centers = []

    drawing_img = np.zeros((FRAME_HEIGHT,FRAME_WIDTH,3),np.uint8)
    drawing_img = cv2.cvtColor(drawing_img,cv2.cv.CV_BGR2GRAY)

    # then draw it
    try:
        #p = cv2.approxPolyDP(contours, 45, False)
        approxContours = []

        for cont in contours:
            try:
                epsilion = 0.03*cv2.arcLength(cont,True) #change the float to change the matching accuracy
                length = length + epsilion/0.03
                p = cv2.approxPolyDP(cont,epsilion,False)
                if(len(p) > 4 and len(p) < 8 and cv2.contourArea(p) > 1000):
                    print(cv2.contourArea(p))
                    x,y,w,h = cv2.boundingRect(p) #get the bounding rectangle
                    #cv2.rectangle(color_img,(x,y),(x+w,y+h),(0,255,0),2)
                    heights.append(h)
                    centers.append(((x+x+w)/2,(y+y+h)/2))
                    approxContours.append(p)
            except TypeError:
                print str(e)
        cv2.drawContours(drawing_img, approxContours, -1, 255, thickness=2) #original second argument = p
    except TypeError as e:
        print str(e)

    #now, do a feature match and see if it is found
    kp2, des2 = orb.detectAndCompute(drawing_img,None)
    try:
        matches = bf.match(des1,des2)
        matchcount = len(matches)
    except:
        matchcount = 0

    #try:
        #print('Distance: '+str(calculate_distance(max(heights))))
    #except ValueError as e:
        #print 'Distance: ERR'
        #print str(e)

    #print(str(centers))

    try:
        leftcenter = centers[0]
        rightcenter = centers[1]

        if leftcenter[0] > rightcenter[0]:
            #switch them
            temp = leftcenter
            leftcenter = rightcenter
            rightcenter = temp
            #now we're correct

        midpoint = ((leftcenter[0]+rightcenter[0])/2)
        gracezone = 40

        if matchcount > 4:

            if midpoint < (SCREEN_MIDPOINT - gracezone):
                print('L')
            elif midpoint > (SCREEN_MIDPOINT + gracezone):
                print('R')
            else:
                print('P')
                #color_image = cv2.cvtColor(color_image,cv2.CV_BGR2GRAY)
        else:
            print('N')
    except:
        pass



    cv2.imshow('Final',morphed_img)
    cv2.imshow('Drawing',drawing_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
