import numpy as np
from cv2 import cv2
from matplotlib import pyplot as plt
from PIL import Image, ImageChops
import time
import random
import string
from os import listdir
import glob

from imageai.Detection import ObjectDetection


def capturarFoto():
	cap = cv2.VideoCapture(0)

	'''leido, frame = cap.read()
	if leido == True:
    	cv2.imwrite("C:\\Users\\USUARIO\\Desktop\\iglue\\foto.png", frame)
    	print("Foto tomada correctamente")
	else:
    	print("Error al acceder a la cámara")
	cap.release()
	cv2.destroyAllWindows()'''

	# Llamada al método
	fgbg = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400, detectShadows=False)

	# Deshabilitamos OpenCL, si no hacemos esto no funciona
	cv2.ocl.setUseOpenCL(False)
	cont = 0
	while(1):
		# Leemos el siguiente frame
		ret, frame = cap.read()
		ret, var = cap.read()

		# Si hemos llegado al final del vídeo salimos
		if not ret:
			break
 
		# Aplicamos el algoritmo
		fgmask = fgbg.apply(frame)
 
		# Copiamos el umbral para detectar los contornos
		contornosimg = fgmask.copy()
 
		# Buscamos contorno en la imagen
		contornos, hierarchy = cv2.findContours(contornosimg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
 
		# Recorremos todos los contornos encontrados
		for c in contornos:
			# Eliminamos los contornos más pequeños
			if cv2.contourArea(c) < 500:
				continue
 
			# Obtenemos el bounds del contorno, el rectángulo mayor que engloba al contorno
			(x, y, w, h) = cv2.boundingRect(c)
		 	#Dibujamos el rectángulo del bounds
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
 
		# Mostramos las capturas
		cv2.imshow('Camara',frame)
		cv2.imshow('gg',var)
		cv2.imshow('Umbral',fgmask)
		cv2.imshow('Contornos',contornosimg)
 
		# Sentencias para salir, pulsa 's' y sale
		k = cv2.waitKey(30) & 0xff
		if k == ord("s"):
			break
 
		# Liberamos la cámara y cerramos todas las ventanas
	
		if cv2.contourArea(c) > 20:
		
			cont = cont +1
			#time.sleep(3)
			cv2.imwrite("/home/juan-rios/Documentos/python/trackMove/original/foto.png", var, time.sleep(0.0))
			print(cont)
			print("foto tomada exitosamente")
			time.sleep(1)
		else:
			print("no se pudo acceder a la camara")
	cap.release()
	cv2.destroyAllWindows()
capturarFoto()
def quitarFondo():
    cap = cv2.imread(r'/home/juan-rios/Documentos/python/trackMove/original/foto.png', 1)
    newImg = cv2.resize(cap, (550, 350))
    print(cap.shape)
    panel = np.zeros([650, 1120], np.uint8)
    cv2.namedWindow('panel')
    def nothing(x):
        pass
    cv2.createTrackbar('L - h', 'panel', 0, 179, nothing)
    cv2.createTrackbar('U - h', 'panel', 179, 179, nothing)
    cv2.createTrackbar('L - s', 'panel', 0, 255, nothing)
    cv2.createTrackbar('U - s', 'panel', 255, 255, nothing)
    cv2.createTrackbar('L - v', 'panel', 0, 255, nothing)
    cv2.createTrackbar('U - v', 'panel', 255, 255, nothing)
    cv2.createTrackbar('S ROWS', 'panel', 0, 480, nothing)
    cv2.createTrackbar('E ROWS', 'panel', 480, 480, nothing)
    cv2.createTrackbar('S COL', 'panel', 0, 640, nothing)
    cv2.createTrackbar('E COL', 'panel', 640, 640, nothing)
    while True:

        frame = newImg[0:650, 0:1120]
        s_r = cv2.getTrackbarPos('S ROWS', 'panel')
        e_r = cv2.getTrackbarPos('E ROWS', 'panel')
        s_c = cv2.getTrackbarPos('S COL', 'panel')
        e_c = cv2.getTrackbarPos('E COL', 'panel')
        roi = frame[s_r: e_r, s_c: e_c]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        l_h = cv2.getTrackbarPos('L - h', 'panel')
        u_h = cv2.getTrackbarPos('U - h', 'panel')
        l_s = cv2.getTrackbarPos('L - s', 'panel')
        u_s = cv2.getTrackbarPos('U - s', 'panel')
        l_v = cv2.getTrackbarPos('L - v', 'panel')
        u_v = cv2.getTrackbarPos('U - v', 'panel')
        lower_green = np.array([l_h, l_s, l_v])
        upper_green = np.array([u_h, u_s, u_v])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        mask_inv = cv2.bitwise_not(mask)
        bg = cv2.bitwise_and(roi, roi, mask=mask)
        fg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        cv2.imshow('filtro', bg)
        cv2.imshow('camara', fg)
        #cv2.imshow('panel', panel)
        cv2.imwrite(r'/home/juan-rios/Documentos/python/trackMove/sin_fondo/foto_sin_fondo.png', fg)
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()

def ponerTransparente():
    src = cv2.imread(r'/home/juan-rios/Documentos/python/trackMove/sin_fondo/foto_sin_fondo.png', 1)
    tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
    b, g, r = cv2.split(src)
    rgba = [b,g,r, alpha]
    dst = cv2.merge(rgba,4)
    size = 15
    chars = string.ascii_uppercase
    cv2.imwrite(r'/home/juan-rios/Documentos/python/trackMove/transparente/trasparente'+  ''.join(random.choice(chars) for _ in range(size))+".png",dst)

def main():
    while r"/home/juan-rios/Documentos/python/trackMove/original/foto.png":
        quitarFondo()
        break
        return 0
    while r"/home/juan-rios/Documentos/python/trackMove/sin_fondo/foto_sin_fondo.png":
        ponerTransparente()
        break
        return 0


        
main()

def final():
	#for cosa in listdir("/Users/jior/Desktop/iglue/trasparente"):
		#print(cosa)

    cv_img = []
    for img in glob.glob("/home/juan-rios/Documentos/python/trackMove/transparente/*.png"):
        n= cv2.imread(img)
        cv_img.append(n)
    #images = [cv2.imread(file) for file in glob.glob('/Users/jior/Desktop/iglue/trasparente/*.png')]
    background = Image.open("/home/juan-rios/Documentos/python/trackMove/original/background.png")
    foreground = Image.open(img)
    size = 128, 128
    background.paste(foreground, (0,0))
    background.show()
final()


'''img = cv2.imread('C:\\Users\\USUARIO\\Desktop\\iglue\\foto.png')
mask = np.zeros(img.shape[:2],np.uint8)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
rect = (50,50,450,290)
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]
plt.imshow(img),plt.colorbar(),plt.show()

# newmask es la máscara etiquetada manualmente
newmask = cv2.imread('C:\\Users\\USUARIO\\Desktop\\iglue\\foto.png',0)
# donde sea que esté marcado en blanco (primer plano seguro), cambiar mask=1
# donde sea que esté marcado en negro (fondo seguro), cambiar mask=0
mask[newmask == 0] = 0
mask[newmask == 255] = 1
mask, bgdModel, fgdModel = cv2.grabCut(img,mask,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask[:,:,np.newaxis]
plt.imshow(img),plt.colorbar(),plt.show()'''