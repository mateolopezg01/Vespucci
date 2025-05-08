# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import csv
# import pandas as pd

# # --- Definición del diccionario ArUco ---
# ARUCO_DICT = {
#     "DICT_4x4_50": cv2.aruco.DICT_4X4_50,
#     "DICT_4x4_100": cv2.aruco.DICT_4X4_100,
#     "DICT_4x4_250": cv2.aruco.DICT_4X4_250,
#     "DICT_4x4_1000": cv2.aruco.DICT_4X4_1000,
#     "DICT_5x5_50": cv2.aruco.DICT_5X5_50,
#     "DICT_5x5_100": cv2.aruco.DICT_5X5_100,
#     "DICT_5x5_250": cv2.aruco.DICT_5X5_250,
#     "DICT_5x5_1000": cv2.aruco.DICT_5X5_1000,
#     "DICT_6x6_50": cv2.aruco.DICT_6X6_50,
#     "DICT_6x6_100": cv2.aruco.DICT_6X6_100,
#     "DICT_6x6_250": cv2.aruco.DICT_6X6_250,
#     "DICT_6x6_1000": cv2.aruco.DICT_6X6_1000,
#     "DICT_7x7_50": cv2.aruco.DICT_7X7_50,
#     "DICT_7x7_100": cv2.aruco.DICT_7X7_100,
#     "DICT_7x7_250": cv2.aruco.DICT_7X7_250,
#     "DICT_7x7_1000": cv2.aruco.DICT_7X7_1000,
#     "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
#     "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
#     "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
#     "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
#     "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
# }

# # --- Función para detectar marcadores ArUco ---
# def detectMarkers(frame, arucoDict="DICT_4x4_250"):
#     dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[arucoDict])
#     parameters = cv2.aruco.DetectorParameters()
#     detector = cv2.aruco.ArucoDetector(dictionary, parameters)
#     corners, ids, rejected = detector.detectMarkers(frame)
#     markers = []
#     if ids is not None:
#         for i in range(len(ids)):
#             marker = {
#                 'id': int(ids[i][0]),
#                 'corners': corners[i][0]  # (4,2) con las 4 esquinas
#             }
#             markers.append(marker)
#     return markers

# # --- Función para construir el conjunto de coordenadas reales (W) ---
# def build_W():
#     """
#     Crea un arreglo W de dimensiones (120,3) con las coordenadas reales del patrón.
#     Para n < 80 se asume z = 0; para n en [80,100) y [100,120) se usan otras fórmulas.
#     """
#     W = np.zeros((120, 3))
#     for n in range(120):
#         if n < 80:
#             W[n] = np.array([(n // 8) * 60, 102 + (n % 8) * 50, 0])
#         elif n < 100: 
#             W[n] = np.array([50 * ((n - 80) // 5), 0, 27 + 60 * ((n - 80) % 5)])
#         else:
#             W[n]=np.array([621,95+50*((n-100)%5),50+60*((n-100)//5)])
#             # W[n] = np.array([400 + 50 * ((n - 100) // 5), 0, 27 + 60 * ((n - 100) % 5)])
#     origin = W[96] + np.array([105, 35, 0])
#     return W, origin

# # --- Función para extraer correspondencias (pRefs y W) ---
# def get_correspondences(imgL, imgR, P1, P2):
#     """
#     Detecta marcadores en ambas imágenes, triangula para obtener los puntos 3D (pRefs)
#     y asocia las coordenadas reales (W) usando el índice del marcador.
#     Se generan dos conjuntos: uno con *todos* los marcadores en común y otro filtrado.
#     """
#     markersL = detectMarkers(imgL)
#     markersR = detectMarkers(imgR)
#     idsL = {m["id"] for m in markersL}
#     idsR = {m["id"] for m in markersR}
#     common_ids = sorted(list(idsL.intersection(idsR)))
    
#     # Construir W completo (coordenadas reales)
#     W_full, origin = build_W()
    
#     # Usar solo aquellos IDs < 120 (dado que W tiene 120 filas)
#     common_ids_all = [i for i in common_ids if i < 120]
    
#     # Conjunto "All": Recorre los marcadores con IDs comunes
#     pRefs_all = []
#     W_all = []
#     markersL_sorted = sorted([m for m in markersL if m["id"] in common_ids_all], key=lambda m: m["id"])
#     markersR_sorted = sorted([m for m in markersR if m["id"] in common_ids_all], key=lambda m: m["id"])
#     for mL, mR in zip(markersL_sorted, markersR_sorted):
#         centerL = np.mean(mL["corners"], axis=0)
#         centerR = np.mean(mR["corners"], axis=0)
#         pts4D = cv2.triangulatePoints(P1, P2, centerL.reshape(2, 1), centerR.reshape(2, 1))
#         pts3D = (pts4D / pts4D[3])[:3].flatten()
#         pRefs_all.append(pts3D)
#         W_all.append(W_full[mL["id"]])
#     pRefs_all = np.array(pRefs_all)
#     W_all = np.array(W_all)
    
#     # Conjunto "Filtered": Filtrar los marcadores con z = 0 (IDs < 80) para obtener datos más robustos.
#     common_ids_z0 = [i for i in common_ids_all if i < 80]
#     common_ids_non_z0 = [i for i in common_ids_all if (i >= 80 and i < 120)]
#     if len(common_ids_z0) > 10:
#         target_positions = []
#         if 26 in common_ids_z0:
#             target_positions.append(W_full[26])
#         if 34 in common_ids_z0:
#             target_positions.append(W_full[34])
#         if len(target_positions) > 0:
#             target = np.mean(target_positions, axis=0)
#         else:
#             target = np.mean([W_full[i] for i in common_ids_z0], axis=0)
#         common_ids_z0 = sorted(common_ids_z0, key=lambda i: np.linalg.norm(W_full[i] - target))[:10]
#     final_ids = sorted(common_ids_non_z0 + common_ids_z0)
    
#     pRefs_filtered = []
#     W_filtered = []
#     markersL_filtered = sorted([m for m in markersL if m["id"] in final_ids], key=lambda m: m["id"])
#     markersR_filtered = sorted([m for m in markersR if m["id"] in final_ids], key=lambda m: m["id"])
#     for mL, mR in zip(markersL_filtered, markersR_filtered):
#         centerL = np.mean(mL["corners"], axis=0)
#         centerR = np.mean(mR["corners"], axis=0)
#         pts4D = cv2.triangulatePoints(P1, P2, centerL.reshape(2, 1), centerR.reshape(2, 1))
#         pts3D = (pts4D / pts4D[3])[:3].flatten()
#         pRefs_filtered.append(pts3D)
#         W_filtered.append(W_full[mL["id"]])
#     pRefs_filtered = np.array(pRefs_filtered)
#     W_filtered = np.array(W_filtered)
    
#     return {"all": (pRefs_all, W_all), "filtered": (pRefs_filtered, W_filtered), "origin": origin}

# def obtTyb(pRefs, W):
#     t = len(W)
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
#     return T, b
# # --- Función para estimar la transformación afín y calcular errores ---
# def estimate_affine_transform(pRefs, W):
#     """
#     Usa cv2.estimateAffine3D para estimar la transformación afín 3D → 3D.
#     Retorna la matriz completa 3x4, la parte lineal (3x3), el vector de traslación,
#     el error medio y máximo, los puntos transformados y los residuals.
#     """
#     # retval, T_full, inliers = cv2.estimateAffine3D(pRefs, W,ransacThreshold=1000)
#     # if retval == 0:
#     #     raise RuntimeError("No se pudo estimar una transformación afín.")
#     # T = T_full[:, :3]    # Parte lineal (3x3)
#     # b = T_full[:, 3]     # Vector de traslación (3,)
#     T_full=0
#     T, b= obtTyb(pRefs,W)
#     predicted = (pRefs @ T.T) + b  # Aplicamos la transformación
#     residuals = np.linalg.norm(predicted - W, axis=1)
#     error_mean = np.mean(residuals)
#     error_max = np.max(residuals)
#     return T_full, T, b, error_mean, error_max, predicted, residuals

# # --- Función para la visualización 3D ---
# def plot_3d_points(pRefs, W, predicted, title):
#     """
#     Grafica en 3D los puntos triangulados (pRefs), las posiciones reales (W)
#     y los puntos transformados (predicted).
#     """
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(pRefs[:,0], pRefs[:,1], pRefs[:,2], c='blue', label='Triangulados (pRefs)')
#     ax.scatter(W[:,0], W[:,1], W[:,2], c='red', marker='^', label='Reales (W)')
#     ax.scatter(predicted[:,0], predicted[:,1], predicted[:,2], c='green', marker='x', label='Transformados')
#     ax.set_title(title)
#     ax.legend()
#     plt.show()

# # --- Función main ---
# def main():
#     # Se cargan las imágenes de prueba (actualiza las rutas según corresponda)
#     imgL = cv2.imread("imgL.png")
#     imgR = cv2.imread("imgR.png")
#     if imgL is None or imgR is None:
#         print("Error al cargar las imágenes de prueba.")
#         return

#     calib_height = 960
#     try:
#         npz_file = np.load(f"./calibration_data960/{calib_height}p/stereo_camera_calibration.npz")
#         if ("leftMapX" in npz_file.files and "leftMapY" in npz_file.files and 
#             "rightMapX" in npz_file.files and "rightMapY" in npz_file.files and 
#             "leftProjection" in npz_file.files and "rightProjection" in npz_file.files):
#             print("Camera calibration data has been found in cache.")
#             mapxL = npz_file["leftMapX"]
#             mapyL = npz_file["leftMapY"]
#             mapxR = npz_file["rightMapX"]
#             mapyR = npz_file["rightMapY"]
#             P1 = npz_file["leftProjection"]
#             P2 = npz_file["rightProjection"]
#             lmtx = npz_file["leftCameraMatrix"]
#             ldst = npz_file["leftDistortionCoeff"]
#         else:
#             print("Camera data file found but data corrupted.")
#             exit(0)
#     except Exception as e:
#         print("Camera calibration data not found in cache.")
#         exit(0)
    
#     # Extraer los puntos de correspondencia para "todos" y para "filtrados"
#     corr = get_correspondences(imgL, imgR, P1, P2)
#     pRefs_all, W_all = corr["all"]
#     pRefs_filt, W_filt = corr["filtered"]
    
#     sets = {"All Markers": (pRefs_all, W_all), "Filtered Markers": (pRefs_filt, W_filt)}
#     results = {}  # Para almacenar resultados y errores
    
#     # Para cada conjunto de puntos, calcular la transformación y los errores usando cv2.estimateAffine3D.
#     for set_name, (pRefs, W_set) in sets.items():
#         T_full, T, b, error_mean, error_max, predicted, residuals = estimate_affine_transform(pRefs, W_set)
#         key = f"{set_name} - Affine3D"
#         results[key] = {
#             "T_full": T_full,
#             "T": T,
#             "b": b,
#             "error_mean": error_mean,
#             "error_max": error_max,
#             "predicted": predicted,
#             "pRefs": pRefs,
#             "W": W_set,
#             "residuals": residuals
#         }
#         print(f"Resultados para {key}:")
#         print("Error medio:", error_mean)
#         print("Error máximo:", error_max)
#         # Visualización 3D de los puntos
#         plot_3d_points(pRefs, W_set, predicted, title=key)
    
#     # Mostrar en consola la comparación de errores para los dos casos
#     for key, res in results.items():
#         print(f"{key}: Error medio = {res['error_mean']:.2f}, Error máximo = {res['error_max']:.2f}")

# if __name__ == "__main__":
#     main()
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import csv
import pandas as pd

# --- Definición del diccionario ArUco ---
ARUCO_DICT = {
    "DICT_4x4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4x4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4x4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4x4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5x5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5x5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5x5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5x5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6x6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6x6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6x6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6x6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7x7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7x7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7x7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7x7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

# --- Función para detectar marcadores ArUco ---
def detectMarkers(frame, arucoDict="DICT_4x4_250"):
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[arucoDict])
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    corners, ids, rejected = detector.detectMarkers(frame)
    markers = []
    if ids is not None:
        for i in range(len(ids)):
            marker = {
                'id': int(ids[i][0]),
                'corners': corners[i][0]  # (4,2) con las 4 esquinas
            }
            markers.append(marker)
    return markers

# Función con refinamiento cv2.cornersSubPix
# def detectMarkers(frame, arucoDict="DICT_4x4_250"):
#     # Convertir la imagen a escala de grises para el refinamiento subpixel
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[arucoDict])
#     parameters = cv2.aruco.DetectorParameters()
#     detector = cv2.aruco.ArucoDetector(dictionary, parameters)
#     corners, ids, rejected = detector.detectMarkers(frame)
#     markers = []
#     if ids is not None:
#         # Definir criterios para el refinamiento subpixel
#         criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
#         for i in range(len(ids)):
#             # Las esquinas originales tienen forma (4, 2)
#             original_corners = corners[i][0]
#             # Remodelar las esquinas a (num_corners, 1, 2) y convertir a float32
#             corners_reshaped = original_corners.reshape(-1, 1, 2).astype(np.float32)
#             # Aplicar el refinamiento subpixel usando la imagen en escala de grises
#             refined_corners = cv2.cornerSubPix(gray, corners_reshaped, (5, 5), (-1, -1), criteria)
#             marker = {
#                 'id': int(ids[i][0]),
#                 # Guardamos las esquinas refinadas en forma (4,2)
#                 'corners': refined_corners.reshape(-1, 2)
#             }
#             markers.append(marker)
#     return markers

# --- Función para construir el conjunto de coordenadas reales (W) ---
def build_W():
    """
    Crea un arreglo W de dimensiones (120,3) con las coordenadas reales del patrón.
    Para n < 80 se asume z = 0; para n en [80,100) y [100,120) se usan otras fórmulas.
    """
    A4_w=210
    sep_x=200
    margin=10
    line=40
    back_margin=27
    W = np.zeros((120, 3))
    for n in range(120):
        if n < 80:
            W[n] = np.array([back_margin+(n // 8) * 60,2+ line+margin + (n % 8) * 50, 0])
        elif n < 100: 
            W[n] = np.array([margin+50 * ((n - 80) % 4), 0, line+margin + 60 * ((n - 80) // 4)])
        else:
            # W[n] = np.array([621, 95 + 50 * ((n - 100) % 5), 50 + 60 * ((n - 100) // 5)])
            W[n] = np.array([A4_w+sep_x+margin + 50 * ((n - 100) % 4), 0, line+margin + 60 * ((n - 100) // 4)])
    origin = W[96] + np.array([105, 35, 0])
    return W, origin

# --- Función para extraer correspondencias (pRefs y W) ---
def get_correspondences(imgL, imgR, P1, P2):
    """
    Detecta marcadores en ambas imágenes rectificadas, triangula para obtener los puntos 3D (pRefs)
    y asocia las coordenadas reales (W) usando el índice del marcador.
    Se generan dos conjuntos: uno con *todos* los marcadores en común y otro filtrado.
    """
    markersL = detectMarkers(imgL)
    markersR = detectMarkers(imgR)
    idsL = {m["id"] for m in markersL}
    idsR = {m["id"] for m in markersR}
    common_ids = sorted(list(idsL.intersection(idsR)))
    
    # Construir W completo (coordenadas reales)
    W_full, origin = build_W()
    
    # Usar solo aquellos IDs < 120 (dado que W tiene 120 filas)
    common_ids_all = [i for i in common_ids if i < 120]
    
    # Conjunto "All": Recorre los marcadores con IDs comunes
    pRefs_all = []
    W_all = []
    markersL_sorted = sorted([m for m in markersL if m["id"] in common_ids_all], key=lambda m: m["id"])
    markersR_sorted = sorted([m for m in markersR if m["id"] in common_ids_all], key=lambda m: m["id"])
    for mL, mR in zip(markersL_sorted, markersR_sorted):
        idL = mL["id"]
        
        corner_idx = 0  # superior izquierda

        cornerL = mL["corners"][corner_idx]
        cornerR = mR["corners"][corner_idx]

        pts4D = cv2.triangulatePoints(P1, P2, cornerL.reshape(2, 1), cornerR.reshape(2, 1))
        pts3D = (pts4D / pts4D[3])[:3].flatten()
        pRefs_all.append(pts3D)
        W_all.append(W_full[idL])

    pRefs_all = np.array(pRefs_all)
    W_all = np.array(W_all)
    
    # Conjunto "Filtered": Filtrar los marcadores con z = 0 (IDs < 80) para obtener datos más robustos.
    common_ids_z0 = [i for i in common_ids_all if i < 80]
    common_ids_non_z0 = [i for i in common_ids_all if (i >= 80 and i < 120)]
    if len(common_ids_z0) > 10:
        target_positions = []
        if 26 in common_ids_z0:
            target_positions.append(W_full[26])
        if 34 in common_ids_z0:
            target_positions.append(W_full[34])
        if len(target_positions) > 0:
            target = np.mean(target_positions, axis=0)
        else:
            target = np.mean([W_full[i] for i in common_ids_z0], axis=0)
        common_ids_z0 = sorted(common_ids_z0, key=lambda i: np.linalg.norm(W_full[i] - target))[:10]
    final_ids = sorted(common_ids_non_z0 + common_ids_z0)
    print(final_ids)
    pRefs_filtered = []
    W_filtered = []
    markersL_filtered = sorted([m for m in markersL if m["id"] in final_ids], key=lambda m: m["id"])
    markersR_filtered = sorted([m for m in markersR if m["id"] in final_ids], key=lambda m: m["id"])
    for mL, mR in zip(markersL_filtered, markersR_filtered):
        idL = mL["id"]
        
        corner_idx = 0  # superior izquierda

        cornerL = mL["corners"][corner_idx]
        cornerR = mR["corners"][corner_idx]

        pts4D = cv2.triangulatePoints(P1, P2, cornerL.reshape(2, 1), cornerR.reshape(2, 1))
        pts3D = (pts4D / pts4D[3])[:3].flatten()
        pRefs_filtered.append(pts3D)
        W_filtered.append(W_full[idL])
    pRefs_filtered = np.array(pRefs_filtered)
    W_filtered = np.array(W_filtered)
    
    return {"all": (pRefs_all, W_all), "filtered": (pRefs_filtered, W_filtered), "origin": origin}

def obtTyb(pRefs, W):
    t = len(W)
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
    return T, b

# --- Función para estimar la transformación afín y calcular errores ---
def estimate_affine_transform(pRefs, W):
    """
    Aquí se usa el método de mínimos cuadrados implementado en obtTyb.
    Se calcula la transformación afín (aunque se podría usar cv2.estimateAffine3D,
    en este ejemplo se utiliza el método manual debido a mejores resultados en Z).
    """
    # retval, T_full, inliers = cv2.estimateAffine3D(pRefs, W)#,ransacThreshold=1000)
    # if retval == 0:
    #     raise RuntimeError("No se pudo estimar una transformación afín.")
    # T = T_full[:, :3]    # Parte lineal (3x3)
    # b = T_full[:, 3]     # Vector de traslación (3,)
    T_full = 0
    T, b = obtTyb(pRefs, W)
    predicted = (pRefs @ T.T) + b  # Aplicamos la transformación
    residuals = np.linalg.norm(predicted - W, axis=1)
    error_mean = np.mean(residuals)
    error_max = np.max(residuals)
    return T_full, T, b, error_mean, error_max, predicted, residuals

# --- Función para la visualización 3D ---
def plot_3d_points(pRefs, W, predicted, title):
    """
    Grafica en 3D los puntos triangulados (pRefs), las posiciones reales (W)
    y los puntos transformados (predicted).
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pRefs[:,0], pRefs[:,1], pRefs[:,2], c='blue', label='Triangulados (pRefs)')
    ax.scatter(W[:,0], W[:,1], W[:,2], c='red', marker='^', label='Reales (W)')
    ax.scatter(predicted[:,0], predicted[:,1], predicted[:,2], c='green', marker='x', label='Transformados')
    ax.set_title(title)
    ax.legend()
    plt.show()

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



# --- Función main ---
def main():
    # Cargar imágenes de prueba (actualiza las rutas según corresponda)
    imgL = cv2.imread("imgL.png")
    imgR = cv2.imread("imgR.png")
    if imgL is None or imgR is None:
        print("Error al cargar las imágenes de prueba.")
        return

    calib_height = 960
    try:
        npz_file = np.load(f"./calibration_data960/{calib_height}p/stereo_camera_calibration.npz")
        if ("leftMapX" in npz_file.files and "leftMapY" in npz_file.files and 
            "rightMapX" in npz_file.files and "rightMapY" in npz_file.files and 
            "leftProjection" in npz_file.files and "rightProjection" in npz_file.files):
            print("Camera calibration data has been found in cache.")
            mapxL = npz_file["leftMapX"]
            mapyL = npz_file["leftMapY"]
            mapxR = npz_file["rightMapX"]
            mapyR = npz_file["rightMapY"]
            roiL = npz_file["leftRoi"]
            roiR = npz_file["rightRoi"]
            P1 = npz_file["leftProjection"]
            P2 = npz_file["rightProjection"]
        else:
            print("Camera data file found but data corrupted.")
            exit(0)
    except Exception as e:
        print("Camera calibration data not found in cache.")
        exit(0)
    
    # --- Aplicar rectificación a las imágenes usando los mapas guardados ---
    # Se asume que estos mapas provienen de la función stereoRectify realizada durante la calibración.
    imgL_rect = cv2.remap(imgL, mapxL, mapyL, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    # x, y, we, he = roiL
    # imgL_rect = imgL_rect[y:y+he, x:x+we]
    imgR_rect = cv2.remap(imgR, mapxR, mapyR, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    # x, y, we, he = roiR
    # imgR_rect = imgR_rect[y:y+he, x:x+we]
    cv2.imshow("imgL_rect",imgL_rect)
    cv2.imshow("imgR_rect",imgR_rect)
    # Ahora se procesan las imágenes rectificadas para extraer correspondencias
    corr = get_correspondences(imgL_rect, imgR_rect, P1, P2)
    pRefs_all, W_all = corr["all"]
    pRefs_filt, W_filt = corr["filtered"]
    # ----------------------------
    # W_set = 10*np.vstack(
    # [
    #     [0, 0, 0],
    #     [5, 0, 0],
    #     [0, 5, 0],
    #     [0, 0, 5],
    #     [10, 0, 0],
    #     [0, 10, 0],
    #     [0, 0, 10],
    #     [15, 0, 0],
    #     [0, 15, 0],
    #     [0, 0, 15],
    #     [0, 0, 20]
    # ]
    # )
    # marcarPuntos=1
    # if marcarPuntos:
    #     pRefs = []
    #     for k in W_set:
    #         pRef = refPoints(imgL_rect, imgR_rect, P1, P2, k)
    #         pRefs.append(pRef)
    #     pRefs = np.array(pRefs)
    #     np.savez("./videos/prefs.npz", pRefs)
    # else:
    #     data = np.load("videos/prefs.npz")
    #     pRefs = data[data.files[0]]

    # T_full, T, b, error_mean, error_max, predicted, residuals = estimate_affine_transform(pRefs, W_set)
    # key = f"Affine3D"
    # results= {
    #     "T_full": T_full,
    #     "T": T,
    #     "b": b,
    #     "error_mean": error_mean,
    #     "error_max": error_max,
    #     "predicted": predicted,
    #     "pRefs": pRefs,
    #     "W": W_set,
    #     "residuals": residuals
    # }
    # print(f"Resultados para marcados:")
    # print("Error medio:", error_mean)
    # print("Error máximo:", error_max)
    # plot_3d_points(pRefs, W_set, predicted, title=key)
    # print(f"Marcados: Error medio = {results['error_mean']:.2f}, Error máximo = {results['error_max']:.2f}")
    # --------------------------------------------
    
    sets = {"All Markers": (pRefs_all, W_all), "Filtered Markers": (pRefs_filt, W_filt)}
    results = {}  # Para almacenar resultados y errores
    
    # Para cada conjunto de puntos, se estima la transformación y se calculan los errores
    for set_name, (pRefs, W_set) in sets.items():
        T_full, T, b, error_mean, error_max, predicted, residuals = estimate_affine_transform(pRefs, W_set)
        key = f"{set_name} - Affine3D"
        results[key] = {
            "T_full": T_full,
            "T": T,
            "b": b,
            "error_mean": error_mean,
            "error_max": error_max,
            "predicted": predicted,
            "pRefs": pRefs,
            "W": W_set,
            "residuals": residuals
        }
        print(f"Resultados para {key}:")
        print("Error medio:", error_mean)
        print("Error máximo:", error_max)
        plot_3d_points(pRefs, W_set, predicted, title=key)
    
    # Mostrar la comparación de errores en consola
    for key, res in results.items():
        print(f"{key}: Error medio = {res['error_mean']:.2f}, Error máximo = {res['error_max']:.2f}")

if __name__ == "__main__":
    main()
