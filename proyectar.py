import cv2
import numpy as np
import matplotlib.pyplot as plt
from proyecciones import update

import cv2
import numpy as np
import sys  # Necesario para salir limpiamente

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
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY)
        # Aplica la erosión para eliminar pequeños ruidos
    eroded_img = cv2.erode(binary_img, kernel, iterations=1)

    # Aplica la dilatación para rellenar huecos pequeños
    processed_img = cv2.dilate(eroded_img, kernel, iterations=1)   
    contours, _ = cv2.findContours(processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Lista para almacenar los centros de los círculos
    centers = []
    # Itera sobre cada contorno
    for contour in contours:
        # Calcula el momento del contorno
        M = cv2.moments(contour)
        if M["m00"] != 0:  # Evita la división por cero
            # Calcula el centroide
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centers.append((cx, cy))
    cv2.imshow("Right Camera", processed_img)
    return centers

def obtCoordenada3D(centroL,centroR):
    points_4D_hom = cv2.triangulatePoints(P1, P2, centroL, centroR)
    points_3D = points_4D_hom / points_4D_hom[3]  # Normalize to non-homogeneous coordinates
    points_3D = points_3D[:3].flatten()
    points_3D = T @ points_3D + b
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
    b = b.flatten()
    return T,b

def obt_pRefs(marcarPuntos):
    if marcarPuntos:
        pRefs=[]
        imgL, imgR = captura(cap)
        for k in W:
            pRef = refPoints(imgL, imgR, P1, P2,k)
            pRefs.append(pRef)
        pRefs = np.array(pRefs)
        np.savez("./videos/prefs.npz", pRefs)
    else:
        data = np.load("videos/prefs.npz")
        pRefs = data[data.files[0]]
    return pRefs

def captura(cap):
        # Capture new frames
    ret, frame = cap.read()
    if not ret:
        print("No se recibió frame de la cámara. Saliendo...")
    else:
        # Corto el video en las dos imágenes
        imgL                 = frame[:video_height,0:video_width//2,:]              #Y+H and X+W
        imgR                = frame[0:video_height,video_width//2:video_width,:]    #Y+H and X+W
    return imgL, imgR

import cv2
import numpy as np
import matplotlib.pyplot as plt
from proyecciones import update

# ---------------------- CARGA DE CALIBRACIÓN ---------------------- #
calib_height = 480
try:
    npz_file = np.load("./calibration_data/{}p/stereo_camera_calibration.npz".format(calib_height))
    if (
        "leftMapX" in npz_file.files and
        "leftMapY" in npz_file.files and
        "rightMapX" in npz_file.files and
        "rightMapY" in npz_file.files and
        "leftProjection" in npz_file.files and
        "rightProjection" in npz_file.files
    ):
        print("Datos de calibración encontrados.")
        P1 = npz_file["leftProjection"]
        P2 = npz_file["rightProjection"]
    else:
        print("Archivo de calibración encontrado pero corrupto.")
        exit(0)
except Exception as e:
    print("Datos de calibración no encontrados:", e)
    exit(0)

# ---------------------- CONFIGURACIÓN DE VIDEO ---------------------- #
video_width = 1280
video_height = 960
stream_url = 'udp://169.254.144.155:3000'  # Cambiar según corresponda
cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
if not cap.isOpened():
    print("Error: No se pudo abrir el stream de video.")
    exit()

# ---------------------- PARÁMETROS ADICIONALES ---------------------- #
# Kernel para procesar la imagen (ya se usa en obtCentros)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Definir otros parámetros necesarios para la triangulación
W = np.vstack([
    [0, 0, 0],
    [5, 0, 0],
    [0, 5, 0],
    [0, 0, 5],
    [10, 0, 0],
    [0, 10, 0],
    [0, 0, 10],
    [15, 0, 0],
    [0, 15, 0],
    [0, 0, 15],
    [20, 0, 0],
    [0, 20, 0],
    [0, 0, 20]
])
t=W.shape[0]
puntos = 4
marcarPuntos = 0
# Se obtiene pRefs y luego se calcula la transformación T y b
pRefs = obt_pRefs(marcarPuntos)    # función definida en tu código
T, b = obtTyb(pRefs)               # función definida en tu código

# ---------------------- CAPTURA DE UN ÚNICO FRAME ---------------------- #
ret, frame = cap.read()
if not ret:
    print("No se pudo capturar un frame.")
    cap.release()
    exit(0)

# Dividir la imagen en dos (izquierda y derecha)
imgL = frame[:video_height, 0:video_width//2, :]
imgR = frame[:video_height, video_width//2:video_width, :]

# ---------------------- PROCESAMIENTO DE LAS IMÁGENES ---------------------- #
# Se obtienen los centros detectados en cada imagen usando obtCentros (que debe dejar de mostrar ventanas extras)
centersL = obtCentros(imgL)
centersR = obtCentros(imgR)

if min(len(centersL), len(centersR)) < puntos:
    print("No se detectaron suficientes puntos en una o ambas imágenes.")
    cap.release()
    exit(0)

# Calcular las coordenadas 3D para cada uno de los 'puntos' (se asume que están en el mismo orden)
new_coords = [None] * puntos
for i in range(puntos):
    # obtCoordenada3D utiliza P1, P2, T y b para la triangulación y transformación
    coordenada3D = obtCoordenada3D(centersL[i], centersR[i])
    new_coords[i] = coordenada3D

# Liberar recursos de video y ventanas de OpenCV
cap.release()
cv2.destroyAllWindows()

# ---------------------- GRAFICADO ESTÁTICO DE LAS PROYECCIONES ---------------------- #
# Configurar la figura con 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
update(new_coords, axes)  # Actualiza los subplots con las proyecciones (XY, XZ, YZ)
plt.show()
