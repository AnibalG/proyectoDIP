import cv2
import numpy as np
import glob
import math
def calcular_volumen(ancho, alto):
    radio=float(ancho)/2
    return math.pi*radio*radio*alto

#valores reales del tamaño del papel
REAL_WIDTH=3.80
REAL_HEIGHT=5.00
#imagen completa
img = cv2.imread('img/proporciones.jpg')
#recorte de donde se ubicó el papel
crop_img = img[450:1100, 1550:2000]
row,col,ch = crop_img.shape

blank_image = np.zeros((row,col,3), np.uint8)
blank_image[:,:] = (0,131,255)
#operacion and con fondo de color naranja, para eliminar ruido de otros colores
mask=cv2.bitwise_and(crop_img,blank_image)
#haciendo que el contraste sea mas notable con sharpening
kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
im = cv2.filter2D(mask, -1, kernel)
#filtro mediano para suavizar la imagen y eliminar ruido
f=cv2.medianBlur(im, 9)
#umbralizando
ret,thresh= cv2.threshold(f,217,255,cv2.THRESH_BINARY)
#cambiando a escala de grises
gray = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
#erosionando imagen para obtener un contorno mas preciso
k = np.ones((5,5),np.uint8)
erosion = cv2.erode(gray,k,iterations = 2)
#encontrando los contornos
contours, hierarchy = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

if len(contours) != 0:
    cv2.drawContours(crop_img, contours, -1, 255, 3)

    # encuentro el area mas grande que abarca el contorno
    c = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(c)
    # dibujo un rectángulo
    rkt = cv2.rectangle(crop_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    print (w)
    print (h)

    x_relation=REAL_WIDTH/w #relación centímetro por píxel (ancho)
    y_relation=REAL_HEIGHT/h #relación centímetro por píxel  (alto)

    print (x_relation)
    print (y_relation)

rutas = glob.glob("img/m*.jpg")
for ruta in rutas:
    print(ruta)
    img = cv2.imread(ruta,0)
    y, x = img.shape
    ycenter = int(y/2)
    xcenter = int(x/2)
    kernel = np.ones((9,9),np.uint8)

    cropped = img[ycenter-1055:ycenter+1030, xcenter-450:xcenter+450]
    color =cv2.cvtColor(cropped,cv2.COLOR_GRAY2BGR)
    ret,thresh = cv2.threshold(cropped,75,255,cv2.THRESH_BINARY_INV)


    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    erosion = cv2.erode(closing, kernel, iterations=4)

    contours, hierarchy = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) != 0:
        # draw blue for all areas
        cv2.drawContours(color, contours, -1, 255, 3)

        #find the biggest area
        c = max(contours, key = cv2.contourArea)

        x,y,w,h = cv2.boundingRect(c)
        # rectangle for biggest area
        cv2.rectangle(color,(x,y),(x+w,y+h),(0,255,0),2)

        ancho=w*x_relation
        alto=h*y_relation

        print(calcular_volumen(ancho,alto))
        #Mostrando la imagen reducida en escala
        scale_percent = 30  # percent of original size
        width = int(color.shape[1] * scale_percent / 100)
        height = int(color.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        resized = cv2.resize(color, dim, interpolation=cv2.INTER_AREA)

        cv2.imshow("fin",resized)
        cv2.waitKey(0)
cv2.destroyAllWindows()