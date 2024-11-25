import numpy as np
import cv2
import matplotlib.pyplot as plt

#Patentes 29.4 x 12.9
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

def image_paths():
    paths = []
    for i in range(1,13):
        if i<10:
            num = '0' + str(i)
        else:
            num = str(i)
        img_path = f"imagenes/img{num}.png"
        paths.append(img_path)
    return paths
def get_images(img_paths):
    """
    Recibe una lista con los path de las imagenes a leer y retorna una lista de las imagenes leidas en bgr
    img_paths: Lista con las direcciones de las imagenes
    imagenes: Lista de las imagenes leidas
    """
    imagenes = []
    for path in img_paths:
        imagenes.append(cv2.imread(path))
    return imagenes
def images_gris(imgs_rgb):
    imagenes = []
    for img in imgs_rgb:
        imagenes.append(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))
    return imagenes
def umbralice(imagen_gris, min = 125,max =255, tipo = "THRESH_BINARY"):
    """
    Recibe una imagen en escala de grises y devuelve la imagen umbralizada
    """
    if tipo == "THRESH_BINARY":
        _, imagen_auto_gris = cv2.threshold(imagen_gris, min, max,cv2.THRESH_BINARY)
    else:
        _, imagen_auto_gris = cv2.threshold(imagen_gris, min, max,cv2.THRESH_BINARY_INV)
    return imagen_auto_gris



paths = get_images(image_paths())
grises = images_gris(paths)
th_images=[]
cortadas=[]
for img in range(len(paths)):
    imagen_auto=grises[img]
    # print(img)
    imagen_auto = imagen_auto[110:285,175:450]
    cortadas.append(imagen_auto)
    umbralizada = umbralice(imagen_auto,132,255)
    th_images.append(umbralizada)

    # plt.figure(figsize=(10, 6))
    # plt.imshow(imagen_auto,cmap='gray',vmin=0,vmax=255)
    # plt.show()
    # plt.figure(figsize=(10, 6))

    # plt.imshow(umbralizada,cmap='gray')
    # plt.title("Imagen Umbralizada")
    # plt.show()

diccionario_componentes={}
diccionario_mask = {}
for i in range(len(th_images)):
    connectivity = 8
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(th_images[i], connectivity, cv2.CV_32S)

    diccionario_componentes[i]=[num_labels, labels, stats, centroids]
    output_image = cortadas[i].copy()
    filtered_mask = np.zeros_like(labels, dtype=np.uint8)

    #Muestra todas las componentes conectadas que hay 
    for j in range(1,len(diccionario_componentes[i][2])):  
            x, y, w, h, area = diccionario_componentes[i][2][j]  
            aspect_ratio = w/h
            if area < 300 and area > 10 and np.isclose(aspect_ratio, 1.5 / 3 ,atol=0.7):
                filtered_mask[labels == j] = 255 #diccionario_componentes[i][1][j]
                cv2.rectangle(output_image, (x-5, y-5), (x + w + 5, y + h + 5), (255, 0, 0), 2)
            diccionario_mask[i]=filtered_mask
    # plt.imshow(output_image,cmap='gray')
    # plt.show()

    # plt.imshow(filtered_mask,cmap='gray')
    # plt.title("Componente Conectada")
    # plt.show()


# print()
# plt.imshow(diccionario_mask[0],cmap='gray')
# plt.title("Primeras componentes conectadas")
# plt.show()

for i in range(len(diccionario_mask)):
    plt.imshow(diccionario_mask[i],cmap='gray')
    plt.title("Componentes Conectadas")
    plt.show()

