import cv2
import numpy as np
import glob

rutas = glob.glob("img/*.jpeg")
for ruta in rutas:
    print(ruta)
    img = cv2.imread(ruta,0)
    y, x = img.shape
    ycenter = int(y/2)
    xcenter = int(x/2)
    kernel = np.ones((9,9),np.uint8)

    cropped = img[ycenter-350:ycenter+350, xcenter-200:xcenter+200]
    color =cv2.cvtColor(cropped,cv2.COLOR_GRAY2BGR) 
    ret,thresh = cv2.threshold(cropped,75,255,cv2.THRESH_BINARY_INV)


    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)



    contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) != 0:
        # draw blue for all areas
        cv2.drawContours(color, contours, -1, 255, 3)

        #find the biggest area
        c = max(contours, key = cv2.contourArea)

        x,y,w,h = cv2.boundingRect(c)
        # rectangle for biggest area
        cv2.rectangle(color,(x,y),(x+w,y+h),(0,255,0),2)



    cv2.imshow("cropped", cropped)
    cv2.imshow("thresh", thresh)
    cv2.imshow("close", closing)
    cv2.imshow("rect", color)
    cv2.waitKey(0)
cv2.destroyAllWindows()