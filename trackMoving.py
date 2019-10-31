import numpy as np
from cv2 import cv2
from matplotlib import pyplot as plt
from PIL import Image, ImageChops
import time
import random
import string
import os
import glob

from imageai.Detection import ObjectDetection


def capturarFoto():
	execution_path = os.getcwd()
	cap = cv2.VideoCapture(0)

	img_in = "image.jpg"
	img_out = "imageo.jpg"

	detector = ObjectDetection()
	detector.setModelTypeAsRetinaNet()
	detector.setModelPath(os.path.join(
		execution_path, "/home/juan-rios/Documentos/python/trackMove/resnet50_coco_best_v2.0.1.h5"))
	detector.loadModel(detection_speed='fast')
	ti = time.time()

	# Llamada al método
	fgbg = cv2.createBackgroundSubtractorKNN(
		history=500, dist2Threshold=400, detectShadows=False)

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
		contornos, hierarchy = cv2.findContours(
			contornosimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		# Recorremos todos los contornos encontrados
		for c in contornos:
			# recObject()
			# Eliminamos los contornos más pequeños
			if cv2.contourArea(c) < 500:
				continue

			# Obtenemos el bounds del contorno, el rectángulo mayor que engloba al contorno
			(x, y, w, h) = cv2.boundingRect(c)
		 	#Dibujamos el rectángulo del bounds
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

			return_value, image = cap.read()
			cv2.imwrite(img_in, image)  # save image
			detections = detector.detectObjectsFromImage(input_image=os.path.join(
				execution_path, img_in), output_image_path=os.path.join(execution_path, img_out))
			to = time.time()
			#print(to-ti)
			image = cv2.imread(img_out)
			image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			cv2.imshow('image', image)
			cv2.imshow('Camara', frame)
			for eachObject in detections:
				print(eachObject["name"], " : ", eachObject["percentage_probability"])
				if(eachObject["name"] == "person" and eachObject["percentage_probability"] > 55):
					print("person suspect")
		

		#cv2.imshow('Camara', frame)
		#cv2.imshow('gg', var)
		#cv2.imshow('Umbral', fgmask)
		#cv2.imshow('Contornos', contornosimg)

		if cv2.waitKey(1) & 0xFF == ord('s'):
			break
		
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

	# Mostramos las capturas
	

	# Sentencias para salir, pulsa 's' y sale
	

		# Liberamos la cámara y cerramos todas las ventanas

	if cv2.contourArea(c) > 20:

		cont = cont + 1
			#time.sleep(3)
		cv2.imwrite(
				"/home/juan-rios/Documentos/python/trackMove/original/foto.png", var, time.sleep(0.0))
		print(cont)
		print("foto tomada exitosamente")
		time.sleep(1)
	cap.release()
	cv2.destroyAllWindows()


capturarFoto()

