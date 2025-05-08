# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from graficado2 import graficado2
# from inicializacion import inicializar
# def refPoints(imgL, imgR, P1, P2,p):
#     window_name = f"Imagen izquierda, seleccione el punto {p}"
#     cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

#     # Mover la ventana a la posición deseada (x, y)
#     cv2.moveWindow(window_name, 100, 100)  # Cambia (100, 100) a la posición deseada

#     bboxL = cv2.selectROI(
#         f"Imagen izquierda, seleccione el punto {p}", imgL, False, False
#     )
#     centerL = np.array(
#         [
#             (bboxL[0] + bboxL[0] + bboxL[2]) / 2,
#             (bboxL[1] + bboxL[1] + bboxL[3]) / 2,
#         ]
#     )
#     cv2.destroyAllWindows()
#     window_name2 = f"Imagen derecha , seleccione el punto {p}"
#     cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

#     # Mover la ventana a la posición deseada (x, y)
#     cv2.moveWindow(window_name, 100, 100)  # Cambia (100, 100) a la posición deseada
#     bboxR = cv2.selectROI(
#         f"Imagen derecha , seleccione el punto {p}", imgR, False, False
#     )
#     centerR = np.array(
#         [
#             (bboxR[0] + bboxR[0] + bboxR[2]) / 2,
#             (bboxR[1] + bboxR[1] + bboxR[3]) / 2,
#         ]
#     )
#     cv2.destroyAllWindows()

#     points_4D_hom = cv2.triangulatePoints(P1, P2, centerL, centerR)
#     points_3D = points_4D_hom / points_4D_hom[3]  # Normalize to non-homogeneous coordinates
#     points_3D = points_3D[:3].flatten()
#     return points_3D

# def obtCentros(img):
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     _, binary_img = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)
#         # Aplica la erosión para eliminar pequeños ruidos
#     eroded_img = cv2.erode(binary_img, kernel, iterations=1)

#     # Aplica la dilatación para rellenar huecos pequeños
#     processed_img = cv2.dilate(eroded_img, kernel, iterations=1)   
#     contours, _ = cv2.findContours(processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     # Lista para almacenar los centros de los círculos
#     centers = []
#     # Itera sobre cada contorno
#     for contour in contours:
#         # Calcula el momento del contorno
#         M = cv2.moments(contour)
#         if M["m00"] != 0:  # Evita la división por cero
#             # Calcula el centroide
#             cx = int(M["m10"] / M["m00"])
#             cy = int(M["m01"] / M["m00"])
#             centers.append((cx, cy))
#     # cv2.imshow("Right Camera", processed_img)
#     return centers

# def obtCoordenada3D(centroL,centroR):
#     points_4D_hom = cv2.triangulatePoints(P1, P2, centroL, centroR)
#     points_3D = points_4D_hom / points_4D_hom[3]  # Normalize to non-homogeneous coordinates
#     points_3D = points_3D[:3].flatten()
#     points_3D = T @ points_3D + b
#     return points_3D

# def obtTyb(pRefs):
#     for k in range(t):
#         r = np.zeros((3, 3 * 4))
#         for l in range(3):
#             r[l, l * 4 : (l + 1) * 4 - 1] = pRefs[k, :]
#             r[l, (l + 1) * 4 - 1] = 1
#         if k == 0:
#             y = np.reshape(W[0, :].T, (3, 1))
#             A = r
#         else:
#             y = np.vstack((y, np.reshape(W[k, :].T, (3, 1))))
#             A = np.vstack((A, r))
#     x, res, rank, s = np.linalg.lstsq(A, y, rcond=None)
#     T = np.zeros((3, 3))
#     b = np.zeros((3, 1))
#     for k in range(3):
#         T[k, :] = x[k * 4 : (k + 1) * 4 - 1, 0]
#         b[k, 0] = x[(k + 1) * 4 - 1, 0]
#     b = b.flatten()
#     return T,b

# def captura(cap):
#         # Capture new frames
#     ret, frame = cap.read()
#     if not ret:
#         print("No se recibió frame de la cámara. Saliendo...")
#     else:
#         # Corto el video en las dos imágenes
#         imgL                 = frame[:video_height,0:video_width//2,:]              #Y+H and X+W
#         imgR                = frame[0:video_height,video_width//2:video_width,:]    #Y+H and X+W
#     return imgL, imgR

# def obt_pRefs(marcarPuntos):
#     if marcarPuntos:
#         pRefs=[]
#         imgL, imgR = captura(cap)
#         for k in W:
#             pRef = refPoints(imgL, imgR, P1, P2,k)
#             pRefs.append(pRef)
#         pRefs = np.array(pRefs)
#         np.savez("./videos/prefs.npz", pRefs)
#     else:
#         data = np.load("videos/prefs.npz")
#         pRefs = data[data.files[0]]
#     return pRefs
    
# calib_height=480
# try:
#     npz_file = np.load("./calibration_data/{}p/stereo_camera_calibration.npz".format(calib_height))
#     if (
#         "leftMapX"
#         and "leftMapY"
#         and "rightMapX"
#         and "rightMapY"
#         and "leftProjection"
#         and "rightProjection" in npz_file.files
#     ):
#         print("Camera calibration data has been found in cache.")
#         mapxL = npz_file["leftMapX"]
#         mapyL = npz_file["leftMapY"]
#         mapxR = npz_file["rightMapX"]
#         mapyR = npz_file["rightMapY"]
#         P1 = npz_file["leftProjection"]
#         P2 = npz_file["rightProjection"]
#         lmtx = npz_file["leftCameraMatrix"]
#         ldst = npz_file["leftDistortionCoeff"]
#     else:
#         print("Camera data file found but data corrupted.")
#         exit(0)
# except:
#     print(
#         "Camera calibration data not found in cache, file "
#         + "./calibration_data/{}p/stereo_camera_calibration.npz".format(calib_height)
#     )
#     exit(0)


# # Video Resolution
# video_width = 1280
# video_height = 960
# stream_url = 'udp://169.254.144.155:3000'  # Replace with your actual stream URL
# cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
# if not cap.isOpened():
#     print("Error: Unable to open video stream")
#     exit()
# dt = 1 / 20
# (width, height)=(5,5)
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width, height))
# # W = np.vstack(
# #     [
# #         [0, 0, 0],
# #         [5, 0, 0],
# #         [0, 5, 0],
# #         [0, 0, 5],
# #         [10, 0, 0],
# #         [0, 10, 0],
# #         [0, 0, 10],
# #         [15, 0, 0],
# #         [0, 15, 0],
# #         [0, 0, 15],
# #         [20, 0, 0],
# #         [0, 20, 0],
# #         [0, 0, 20],
# #         [5, 5, 0],
# #         [10, 10, 0],
# #         [15, 15, 0],
# #     ]
# # )
# # t = W.shape[0]
# puntos=4
# # marcarPuntos=0
# imgL, imgR=captura(cap)
# T,b, P1, P2, origin=inicializar(imgL, imgR)
# # pRefs=obt_pRefs(marcarPuntos)
# # # T,b=obtTyb(pRefs)
# # T=np.array([[ 19.47994263  , 0.7599567 , -27.4280199 ],
# #  [ -0.52320432 ,-19.53009334 ,  7.61787927],
# #  [-19.36131188  , 1.43357721 ,-32.89791989]])
# # b=np.array([400.37238794 ,166.73721972, 572.13336392])

# # Coordenadas iniciales vacías
# coords = [[] for _ in range(puntos)]
# coordenadas = []

# # Configurar el objeto VideoWriter para guardar el video
# # output_filename = './videos/salida.avi'  # Cambia el nombre y formato si deseas otro
# # fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Usa un códec como XVID o MJPG
# fps = 20  # Fotogramas por segundo
# frame_size = (video_width, video_height)  # Tamaño del video

# # out = cv2.VideoWriter(output_filename, fourcc, fps, frame_size)

# while True:
#     # Capturar imágenes izquierda y derecha
#     imgL, imgR = captura(cap)

#     # Procesar imágenes
#     centersL = obtCentros(imgL)
#     centersR = obtCentros(imgR)

#     if min(len(centersL), len(centersR)) >= puntos:
#         # Lista temporal para nuevas coordenadas
#         new_coords = [None] * puntos

#         for i in range(puntos):
#             coordenada3D = obtCoordenada3D(centersL[i], centersR[i])
#             # coordenada3D[1]+=40
#             coordenada3D-=origin
#             # graficado(coordenada3D)
#             if len(coords) > 0 and all(isinstance(c, list) and len(c) > 0 for c in coords):
#                 # Buscar la coordenada previa más cercana
#                 prev_coords = np.array(coords)
#                 distances = np.linalg.norm(prev_coords - coordenada3D, axis=1)
#                 min_index = np.argmin(distances)
#                 new_coords[min_index] = coordenada3D
#             else:
#                 new_coords[i] = coordenada3D

#         # Actualizar coordenadas
#         coords = new_coords
#         coordenadas.append(coords)
#         graficado2(coords)

#     # Mostrar las imágenes en ventanas
#     combined_frame = np.hstack((imgL, imgR))  # Combinar las imágenes izquierda y derecha
#     # cv2.imshow('Video Stereo', combined_frame)

#     # Escribir el fotograma combinado en el archivo de salida
#     # out.write(combined_frame)

#     # Salir con la tecla "q"
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# # Liberar recursos
# cap.release()
# # out.release()  # Importante para guardar el archivo
# cv2.destroyAllWindows()

# # Guardar las coordenadas 3D
# coordenadas = np.vstack(coordenadas)
# ax = plt.figure().add_subplot(projection='3d')
# ax.plot(coordenadas[:, 0], coordenadas[:, 1], coordenadas[:, 2], 'k.')
# ax.plot(coordenadas[:, 0], coordenadas[:, 1], coordenadas[:, 2], 'rx')
# ax.plot(coordenadas[-1, 0], coordenadas[-1, 1], coordenadas[-1, 2], 'ro')
# # plt.savefig('./videos/video_40_pts.png')
# plt.show()
import cv2
import numpy as np
import threading
import time
import matplotlib.pyplot as plt
# from graficado import graficado
from graficado2 import graficado2
from inicio2 import inicializar
import sys

class ThreadedVideoCapture:
    def __init__(self, stream_url):
        self.cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            raise Exception("Error: No se pudo abrir el stream")
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                print("Error al capturar frame en el hilo de captura")
                time.sleep(0.01)  # Pequeña espera para evitar busy waiting

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.thread.join()
        self.cap.release()


def get_point(img, window_name, screen_width, screen_height):
    """
    Muestra la imagen en una ventana y espera a que el usuario haga clic para seleccionar un punto.
    Si se presiona 'q', se termina el programa.
    """
    point = []

    # Callback del mouse para capturar el clic
    def mouse_callback(event, x, y, flags, param):
        nonlocal point
        if event == cv2.EVENT_LBUTTONDOWN:
            point = [x, y]
            cv2.destroyWindow(window_name)

    # Configura la ventana
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, screen_width, screen_height)
    cv2.moveWindow(window_name, 0, 0)
    cv2.setMouseCallback(window_name, mouse_callback)
    cv2.imshow(window_name, img)

    # Espera a que se seleccione un punto o se presione 'q'
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Se presionó 'q'. Saliendo del programa...")
            cv2.destroyAllWindows()
            sys.exit()
        if point:
            break

    return np.array(point, dtype=np.float32)

def refPoints(imgL, imgR, P1, P2, p):
    """
    Permite seleccionar un punto en la imagen izquierda y otro en la imagen derecha.
    Luego triangula dichos puntos usando las matrices de proyección P1 y P2 y retorna
    las coordenadas 3D resultantes.
    """
    # Definir tamaño de la ventana (puede ajustarse a la resolución de tu pantalla)
    screen_width = 1920
    screen_height = 1080

    # Seleccionar punto en la imagen izquierda
    window_name_L = f"Imagen izquierda - punto {p}"
    centerL = get_point(imgL, window_name_L, screen_width, screen_height)

    # Seleccionar punto en la imagen derecha
    window_name_R = f"Imagen derecha - punto {p}"
    centerR = get_point(imgR, window_name_R, screen_width, screen_height)

    # Asegurarse de que los puntos tengan la forma (2, 1) y sean de tipo float32
    centerL = centerL.reshape(2, 1)
    centerR = centerR.reshape(2, 1)

    # Triangulación: las matrices P1 y P2 deben ser adecuadas para cv2.triangulatePoints
    points_4D_hom = cv2.triangulatePoints(P1, P2, centerL, centerR)
    points_3D = points_4D_hom / points_4D_hom[3]  # Normalizar a coordenadas no homogéneas
    return points_3D[:3].flatten()

def obtCentros(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    eroded_img = cv2.erode(binary_img, kernel, iterations=1)
    processed_img = cv2.dilate(eroded_img, kernel, iterations=1)
    contours, _ = cv2.findContours(processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    cv2.imshow("messi",processed_img)
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centers.append((cx, cy))
    return centers

def obtCoordenada3D(centroL, centroR):
    centroL = np.array(centroL, dtype=np.float32).reshape(2,1)
    centroR = np.array(centroR, dtype=np.float32).reshape(2,1)
    points_4D_hom = cv2.triangulatePoints(P1, P2, centroL, centroR)
    points_3D = points_4D_hom / points_4D_hom[3]
    points_3D = points_3D[:3].flatten()
    points_3D = T @ points_3D + b
    # points_3D = np.array([reg.predict(points_3D.reshape(1, -1))[0] for reg in regresores])
    return points_3D

def obtTyb(pRefs):
    for k in range(t):
        r = np.zeros((3, 3 * 4))
        for l in range(3):
            r[l, l * 4 : (l + 1) * 4 - 1] = pRefs[k, :]
            r[l, (l + 1) * 4 - 1] = 1
        if k == 0:
            y = np.reshape(W[0, :].T, (3, 1))
            A = r
        else:
            y = np.vstack((y, np.reshape(W[k, :].T, (3, 1))))
            A = np.vstack((A, r))
    x, res, rank, s = np.linalg.lstsq(A, y, rcond=None)
    T = np.zeros((3, 3))
    b = np.zeros((3, 1))
    for k in range(3):
        T[k, :] = x[k * 4 : (k + 1) * 4 - 1, 0]
        b[k, 0] = x[(k + 1) * 4 - 1, 0]
    return T, b.flatten()

def obt_pRefs(marcarPuntos):
    if marcarPuntos:
        pRefs = []
        frame = cap_thread.read()
        if frame is None:
            print("Error al capturar imágenes para pRefs.")
            return None
        imgL = frame[:video_height, 0:video_width//2, :]
        imgR = frame[:video_height, video_width//2:video_width, :]
        for k in W:
            pRef = refPoints(imgL, imgR, P1, P2, k)
            pRefs.append(pRef)
        pRefs = np.array(pRefs)
        np.savez("./videos/prefs.npz", pRefs)
    else:
        data = np.load("videos/prefs.npz")
        pRefs = data[data.files[0]]
    return pRefs


# Parámetros del video y creación del hilo de captura
video_width = 1280
video_height = 960
stream_url = 'udp://169.254.144.155:3000'
try:
    cap_thread = ThreadedVideoCapture(stream_url)
except Exception as e:
    print(e)
    exit(0)

dt = 1 / 20
(width, height) = (5, 5)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width, height))

puntos = 4

# Bucle de reintentos para obtener el primer frame
max_attempts = 10
attempts = 0
initial_frame = None
while attempts < max_attempts and initial_frame is None:
    initial_frame = cap_thread.read()
    if initial_frame is None:
        print(f"Intento {attempts+1}: No se pudo capturar el frame inicial, reintentando...")
        time.sleep(0.1)
        attempts += 1

if initial_frame is None:
    print("Error: No se pudo obtener un frame inicial tras varios intentos. Saliendo.")
    cap_thread.stop()
    exit(0)

# Dividir el frame inicial en imágenes izquierda y derecha para la inicialización
imgL_init = initial_frame[:video_height, 0:video_width//2, :]
imgR_init = initial_frame[:video_height, video_width//2:video_width, :]

# T, b, P1, P2, origin = inicializar(imgL_init, imgR_init)
T,b,P1,P2,origin,(mxL,myL,mxR,myR),_,_, regresores = inicializar(imgL_init, imgR_init)
coords = [[] for _ in range(puntos)]
coordenadas = []

alpha=0.1
# Bucle principal de procesamiento
while True:
    frame = cap_thread.read()
    if frame is None:
        continue  # Si no se ha recibido frame, seguir esperando

    # Dividir el frame en imágenes izquierda y derecha
    imgL = frame[:video_height, 0:video_width//2, :]
    imgR = frame[:video_height, video_width//2:video_width, :]

    imgL=cv2.remap(imgL,mxL,myL,cv2.INTER_LINEAR)
    imgR=cv2.remap(imgR,mxR,myR,cv2.INTER_LINEAR)

    # Procesar imágenes
    centersL = obtCentros(imgL)
    centersR = obtCentros(imgR)

    # ORDENAR LOS CENTROS
    centersL = sorted(centersL, key=lambda c: c[0])
    centersR = sorted(centersR, key=lambda c: c[0])

    if min(len(centersL), len(centersR)) >= puntos:
        new_coords = [None] * puntos
        for i in range(puntos):
            coordenada3D = obtCoordenada3D(centersL[i], centersR[i])
            coordenada3D -= origin
            # graficado(coordenada3D)
            if len(coords) > 0 and all(isinstance(c, list) and len(c) > 0 for c in coords):
                prev_coords = np.array(coords)
                distances = np.linalg.norm(prev_coords - coordenada3D, axis=1)
                min_index = np.argmin(distances)
                # new_coords[min_index] = coordenada3D
                # Filtro exponencial: se pondera la nueva coordenada con la anterior
                smoothed_coord = np.stack([int(alpha * coordenada3D[i] + (1 - alpha) * prev_coords[min_index]) for i in range(3)])
                new_coords[min_index] = smoothed_coord
            else:
                new_coords[i] = coordenada3D

        coords = new_coords
        coordenadas.append(coords)
        print(coords[2])
        graficado2(coords)

    # Si lo deseas, puedes mostrar el frame combinado
    # combined_frame = np.hstack((imgL, imgR))
    # cv2.imshow('Video Stereo', combined_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# con filtro pasabajos

# while True:
#     frame = cap_thread.read()
#     if frame is None:
#         continue  # Esperar si no se recibe frame
    
#     # Dividir el frame en imágenes izquierda y derecha
#     imgL = frame[:video_height, 0:video_width//2, :]
#     imgR = frame[:video_height, video_width//2:video_width, :]

#     # Procesar imágenes para obtener centros (u otras características)
#     centersL = obtCentros(imgL)
#     centersR = obtCentros(imgR)

#     if min(len(centersL), len(centersR)) >= puntos:
#         new_coords = [None] * puntos
#         for i in range(puntos):
#             # Calcular la coordenada 3D para el marcador 'i'
#             coordenada3D = obtCoordenada3D(centersL[i], centersR[i])
            
#             # Ajustar la coordenada respecto al origen
#             coordenada3D -= origin
#             # graficado(coordenada3D)

#             # Aplicar suavizado si ya existen coordenadas previas
#             if len(coords) > 0 and all(isinstance(c, np.ndarray) for c in coords):
#                 prev_coords = np.array(coords)
#                 distances = np.linalg.norm(prev_coords - coordenada3D, axis=1)
#                 min_index = np.argmin(distances)

#             else:
#                 new_coords[i] = coordenada3D

#         # Actualizar las coordenadas para el siguiente ciclo y almacenar para graficar
#         coords = new_coords
        
#         if coords is not None:
#             graficado2(coords)
#             coordenadas.append(coords)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

cap_thread.stop()
cv2.destroyAllWindows()

# Guardar y graficar las coordenadas 3D
if coordenadas:
    coordenadas = np.vstack(coordenadas)
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(coordenadas[:, 0], coordenadas[:, 1], coordenadas[:, 2], 'k.')
    ax.plot(coordenadas[:, 0], coordenadas[:, 1], coordenadas[:, 2], 'rx')
    ax.plot(coordenadas[-1, 0], coordenadas[-1, 1], coordenadas[-1, 2], 'ro')
    plt.show()
else:
    print("No se han capturado coordenadas 3D.")

