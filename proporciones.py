import cv2 as cv
import numpy as np
#valores reales del tamaño del papel
REAL_WIDTH=3.80
REAL_HEIGHT=5.00
#imagen completa
img = cv.imread('img/proporciones.jpg')
#recorte de donde se ubicó el papel
crop_img = img[450:1100, 1550:2000]
row,col,ch = crop_img.shape

blank_image = np.zeros((row,col,3), np.uint8)
blank_image[:,:] = (0,131,255)
#operacion and con fondo de color naranja, para eliminar ruido de otros colores
mask=cv.bitwise_and(crop_img,blank_image)
#haciendo que el contraste sea mas notable con sharpening
kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
im = cv.filter2D(mask, -1, kernel)
#filtro mediano para suavizar la imagen y eliminar ruido sal-pimienta
f=cv.medianBlur(im, 9)
#umbralizando
ret,thresh= cv.threshold(f,217,255,cv.THRESH_BINARY)
#cambiando a escala de grises
gray = cv.cvtColor(thresh, cv.COLOR_BGR2GRAY)
#erosionando imagen para obtener un contorno mas preciso
k = np.ones((5,5),np.uint8)
erosion = cv.erode(gray,k,iterations = 2)
#encontrando los contornos
contours, hierarchy = cv.findContours(erosion, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

if len(contours) != 0:
    cv.drawContours(crop_img, contours, -1, 255, 3)

    # encuentro el area mas grande que abarca el contorno
    c = max(contours, key=cv.contourArea)

    x, y, w, h = cv.boundingRect(c)
    # dibujo un rectángulo
    rkt = cv.rectangle(crop_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    x_relation=REAL_WIDTH/w #relación centímetro por píxel (ancho)
    y_relation=REAL_HEIGHT/h #relación centímetro por píxel  (alto)

    print (x_relation)
    print (y_relation)

cv.imshow("0",crop_img)

cv.waitKey(0)
cv.destroyAllWindows()