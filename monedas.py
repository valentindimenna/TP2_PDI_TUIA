import numpy as np
import cv2
import matplotlib.pyplot as plt
#19 objetos en total, 17 monedas 2 dados
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
img_path = "imagenes/monedas.jpg"
# img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

##########Puede ser ###################
img = cv2.imread('imagenes/monedas.jpg')
imghsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
# lower_bound = np.array([50, 3, 180])  
# upper_bound = np.array([100, 15, 250])
lower_bound = np.array([50, 3, 180])  # Limite Inferior de cada Canal
upper_bound = np.array([110, 18, 250])# Limite Superior de cada Canal
mask = cv2.inRange(imghsv, lower_bound, upper_bound)
result = cv2.bitwise_and(img, img, mask=mask)
# imshow(imghsv)
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(mask, cmap="gray")
plt.title("Máscara de Color")
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title("Objetos Segmentados")
plt.subplot(4,3,1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
plt.title("Imagen HSV 50-100 3-18 180-250")
plt.show()

# Encontrar contornos en la imagen binaria
contours, jerarquia = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
filtered_contours = [(i,cnt) for i,cnt in enumerate(contours) if cv2.contourArea(cnt) > 2000]
# print(len(filtered_contours))

dados = []
indices = []
for i,cnt in filtered_contours:
    # Crear un rectángulo delimitador alrededor del contorno
    x, y, w, h = cv2.boundingRect(cnt)
    indices.append(i)
    # Recortar el objeto en la imagen original en color
    objeto = result[y:y+h, x:x+w]
    # Mostrar el objeto
    dados.append(objeto)
    plt.imshow(cv2.cvtColor(objeto, cv2.COLOR_BGR2RGB))
    plt.show()

print("aca hay un print que no anda")
dict_contornos = {}
padres = 0
hijos = 0
conteo_dado = 0
for indice in indices:
    _, _, first_child, parent = jerarquia[0][indice]
    # print(indice,jerarquia[0][indice])
    # print(first_child,parent)
    if parent == -1:
        #Es padre 
        padres +=1 
        color = (0, 0, 255)  # Rojo para contornos padres
        dict_contornos[indice] = []  # Inicializo la lista de hijos para este padre
    else:  
        # Contorno hijo
        hijos += 1
        color = (0, 255, 0)  # Verde para contornos hijos
        # Agregar este contorno hijo a la lista del contorno padre
        if parent in dict_contornos:
            dict_contornos[parent].append(indice)
    conteo_dado = hijos
    cv2.drawContours(img, contours, indice, color, 1)

print("ESTE ES EL QUE ESTA ADENTRO DEL FOR",conteo_dado)
conteo_dados = hijos
print("ESTE ES AFUERA DEL FOR",conteo_dados)

# conteo_dados = hijos Esto tira un ERROR Y NO ENTIENDO PORQUE



######ACA TENGO LOS DADOS
# Mostrar la imagen con contornos dibujados
# print(f' padres: {padres}, hijos : {hijos}')
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Contornos de dados Detectados")
plt.axis('off')
plt.show()

######################MONEDAS
lower_bound = np.array([0, 0, 90])  # Ajusta estos valores
upper_bound = np.array([40, 255, 255])
mask = cv2.inRange(imghsv, lower_bound, upper_bound)
result = cv2.bitwise_and(img, img, mask=mask)
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(mask, cmap="gray")
plt.title("Máscara de Color")
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title("Objetos Segmentados")
plt.subplot(4,3,1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
plt.title("Imagen HSV 5-50 70-200 120-250")
plt.show()
imshow(imghsv)
contours, jerarquia = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
filtered_contours = [(i,cnt) for i,cnt in enumerate(contours) if cv2.contourArea(cnt) > 35000]
print(len(filtered_contours))

monedas = []
indices = []
for i,cnt in filtered_contours:
    # Crear un rectángulo delimitador alrededor del contorno
    x, y, w, h = cv2.boundingRect(cnt)
    indices.append(i)
    # Recortar el objeto en la imagen original en color
    objeto = result[y:y+h, x:x+w]
    
    # Mostrar el objeto
    
    monedas.append(objeto)
    plt.imshow(cv2.cvtColor(objeto, cv2.COLOR_BGR2RGB))
    plt.show()
# cv2.contourArea(contornos[key])
areas = []
m1 = []
m01 = []
m05 = []
for i in range(len(monedas)):
    areas.append(cv2.contourArea(contours[indices[i]]))
    if areas[i] >= 95000:
        m05.append(indices[i])
    elif areas[i] >= 80000:
        m1.append(indices[i])
    else:
        m01.append(indices[i])

print(len(m1)+len(m01)*0.1+len(m05)*0.5)