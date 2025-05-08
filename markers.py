import cv2
import numpy as np

# Elegimos el diccionario ArUco que queremos usar (por ejemplo: DICT_4X4_50)
diccionario = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# Tamaño del marcador (en píxeles)
lado_marcador = 400

# ID del marcador (elige uno desde 0 hasta el máximo permitido por tu diccionario)
id_marcador = 0

# Creamos la imagen del marcador ArUco
marcador_img = np.zeros((lado_marcador, lado_marcador), dtype=np.uint8)
marcador_img = cv2.aruco.generateImageMarker(diccionario, id_marcador, lado_marcador, marcador_img, 1)

# Guardamos el marcador generado
cv2.imwrite("marcador_aruco_0.png", marcador_img)

# Opcional: mostramos el marcador
cv2.imshow('Marcador ArUco', marcador_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
