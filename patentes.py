import numpy as np
import cv2
import matplotlib.pyplot as plt
#12 Imagenes de patentes 
#BUF 817
def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=True, ticks=False):
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:        
        plt.show(block=blocking)
i="03"
imagen_auto = cv2.imread(f"imagenes/img{i}.png")
imagen_auto_rgb = cv2.cvtColor(imagen_auto,cv2.COLOR_BGR2RGB)
imagen_auto_gris = cv2.cvtColor(imagen_auto,cv2.COLOR_BGR2GRAY)
imagen_auto_lab = cv2.cvtColor(imagen_auto,cv2.COLOR_BGR2LAB)
imshow(imagen_auto)

_,threshold = cv2.threshold(imagen_auto_gris,100,255,cv2.THRESH_BINARY)
imshow(threshold)        

contornos,_ = cv2.findContours(threshold,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
coso = np.zeros_like(imagen_auto_rgb)

cv2.drawContours(coso,contornos,-1,(255,0,0),2)

imshow(coso)

gris = cv2.blur(imagen_auto_gris,(3,3))
imshow(gris)
canny = cv2.Canny(gris,150,200)
canny= cv2.dilate(canny,None,iterations = 1)
imshow(canny)
nuevo_cnt = []
cnt,_ = cv2.findContours(canny,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

filtered_contours = [(i,cnt) for i,cnt in enumerate(cnt) if cv2.contourArea(cnt) > 2000000 or cv2.contourArea(cnt)>3000]
len(filtered_contours)
for c in filtered_contours:
    area = cv2.contourArea(c)
    if not(area > 20000 or area<1000):
        nuevo_cnt.append(c)
indices = []
cosito=[]
for i,cnt in filtered_contours:
    # Crear un rectángulo delimitador alrededor del contorno
    x, y, w, h = cv2.boundingRect(cnt)
    indices.append(i)
    # Recortar el objeto en la imagen original en color
    objeto = gris[y:y+h, x:x+w]
    # Mostrar el objeto
    cosito.append(objeto)
    plt.imshow(cv2.cvtColor(objeto, cv2.COLOR_BGR2RGB))
    plt.show()
