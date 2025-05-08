import os
import cv2
import numpy as np

# ================================
# Configuración global y parámetros
# ================================

total_photos = 55

# Resolución completa de la foto (lado a lado: dos cámaras)
photo_width = 1280  
photo_height = 960

# Resolución para procesamiento (cada cámara se procesa con 640x960)
img_width = 640
img_height = 960
image_size = (img_width, img_height)

# Parámetros del tablero de ajedrez
rows = 6
columns = 9
square_size = 28  # Tamaño real de cada cuadrado (por ejemplo, en milímetros)

# Opciones de visualización
drawCorners = False
showSingleCamUndistortionResults = True
showStereoRectificationResults = True
writeUndistortedImages = True  # Se corrige el nombre (era writeUdistortedImages)
imageToDisp = './scenes960/scene_1280x960_1.png'

# Configuración para la calibración
CHECKERBOARD = (rows, columns)
subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = cv2.CALIB_FIX_INTRINSIC

# ----------------------------------------------------------
# Cambio 1: Uso de 'square_size' para escalar los puntos 3D reales
# Se generan los puntos del tablero y se multiplica por square_size
# ----------------------------------------------------------
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

# Listas para almacenar puntos detectados para las cámaras izquierda y derecha
objpointsLeft = []  # Puntos 3D en el mundo real
imgpointsLeft = []  # Puntos 2D en la imagen (izquierda)
objpointsRight = []  # Puntos 3D para la cámara derecha (idénticos al de la izquierda)
imgpointsRight = []  # Puntos 2D en la imagen (derecha)

# =======================================
# Función para procesar las imágenes de calibración
# =======================================
def process_calibration_images():
    """
    Procesa los pares de imágenes de calibración y llena las listas
    de puntos 3D (objpoints) y puntos 2D (imgpoints).
    """
    global total_photos, photo_width, photo_height, img_width, img_height
    photo_counter = 0
    while photo_counter < total_photos:
        photo_counter += 1
        print(f'Procesando par No {photo_counter}')
        leftName = f'./pairs960/left_{photo_counter:02d}.png'
        rightName = f'./pairs960/right_{photo_counter:02d}.png'
        leftExists = os.path.isfile(leftName)
        rightExists = os.path.isfile(rightName)
        
        # Se verifica que ambos archivos existan
        if ((not leftExists) or (not rightExists)) and (leftExists != rightExists):
            print(f"El par No {photo_counter} tiene solo una imagen! Left: {leftExists}, Right: {rightExists}")
            continue
        
        if leftExists and rightExists:
            imgL = cv2.imread(leftName, 1)
            loadedY, loadedX, _ = imgL.shape
            grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
            # Se redimensiona para procesamiento
            gray_small_left = cv2.resize(grayL, (img_width, img_height), interpolation=cv2.INTER_AREA)
            
            imgR = cv2.imread(rightName, 1)
            grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
            gray_small_right = cv2.resize(grayR, (img_width, img_height), interpolation=cv2.INTER_AREA)
            
            # Buscar las esquinas del tablero
            retL, cornersL = cv2.findChessboardCorners(grayL, CHECKERBOARD,
                                                         cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
            retR, cornersR = cv2.findChessboardCorners(grayR, CHECKERBOARD,
                                                         cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
            
            # Visualización opcional de esquinas encontradas
            if drawCorners:
                cv2.drawChessboardCorners(imgL, CHECKERBOARD, cornersL, retL)
                cv2.imshow('Esquinas Izquierda', imgL)
                cv2.drawChessboardCorners(imgR, CHECKERBOARD, cornersR, retR)
                cv2.imshow('Esquinas Derecha', imgR)
                key = cv2.waitKey(0)
                if key == ord("q"):
                    exit(0)
            
            # Filtrar imágenes donde el tablero esté demasiado cerca del borde
            SayMore = False  # Activar para mayor información de depuración
            if retL and retR:
                minRx = cornersR[:, :, 0].min()
                minRy = cornersR[:, :, 1].min()
                minLx = cornersL[:, :, 0].min()
                minLy = cornersL[:, :, 1].min()
                
                border_threshold_x = loadedX / 100
                border_threshold_y = loadedY / 100
                x_thresh_bad = (minRx < border_threshold_x) or (minLx < border_threshold_x)
                y_thresh_bad = (minRy < border_threshold_y) or (minLy < border_threshold_y)
                if x_thresh_bad or y_thresh_bad:
                    if SayMore:
                        print("Tablero demasiado cerca del borde!")
                    else:
                        print("Tablero demasiado cerca del borde! Imagen ignorada")
                    continue
            
            # ----------------------------------------------------------
            # Cambio 2: Escalar esquinas a la resolución de procesamiento
            # ----------------------------------------------------------
            if retL and retR and (img_height <= photo_height):
                scale_ratio = img_height / photo_height
                print(f"Scale ratio: {scale_ratio}")
                cornersL = cornersL * scale_ratio
                cornersR = cornersR * scale_ratio
            elif img_height > photo_height:
                print("La resolución de imagen es mayor que la resolución de la foto. Revisar parámetros.")
                exit(0)
            
            # Refinar esquinas y agregarlas a las listas
            if retL and retR:
                objpointsLeft.append(objp)
                cv2.cornerSubPix(gray_small_left, cornersL, (3, 3), (-1, -1), subpix_criteria)
                imgpointsLeft.append(cornersL)
                objpointsRight.append(objp)
                cv2.cornerSubPix(gray_small_right, cornersR, (3, 3), (-1, -1), subpix_criteria)
                imgpointsRight.append(cornersR)
            else:
                print(f"El par No {photo_counter} se ignora, ya que no se encontró el tablero.")
                continue
    print("Ciclo de procesamiento finalizado")
    # Se retorna además la forma de la imagen (para evitar depender de variables globales)
    return (objpointsLeft, imgpointsLeft, objpointsRight, imgpointsRight, grayL.shape)

# ==============================================
# Función de calibración individual de cada cámara
# ==============================================
def calibrate_one_camera(objpoints, imgpoints, side, image_shape):
    """
    Calibra una sola cámara y guarda los mapas para undistortion.
    Parámetros:
      - objpoints, imgpoints: Listas de puntos 3D y 2D.
      - side: 'left' o 'right'.
      - image_shape: forma de la imagen (alto, ancho) obtenido en el procesamiento.
    Se elimina la dependencia de variables globales usando image_shape.
    """
    DIM = (image_shape[1], image_shape[0])  # (ancho, alto)
    rms, camera_matrix, distortion_coeff, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, DIM, None, None)
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coeff, DIM, 1, DIM)
    map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, distortion_coeff, None, new_camera_mtx, DIM, cv2.CV_16SC2)
    
    # Asegurarse de que el directorio exista
    calib_dir = f'./calibration_data960/{image_shape[0]}p'
    if not os.path.isdir(calib_dir):
        os.makedirs(calib_dir)
    
    # ----------------------------------------------------------
    # Cambio 3: Uso de f-strings para formar nombres de archivo (se evita usar '&')
    # y se corrige la condición de existencia de claves en el npz
    # ----------------------------------------------------------
    np.savez(os.path.join(calib_dir, f'camera_calibration_{side}.npz'),
             map1=map1, map2=map2, objpoints=objpoints, imgpoints=imgpoints,
             camera_matrix=camera_matrix, distortion_coeff=distortion_coeff, roi=roi)
    return (new_camera_mtx, distortion_coeff, roi)

# ==========================================================
# Función de calibración estereoscópica de ambas cámaras
# ==========================================================
def calibrate_stereo_cameras(left_data, right_data, image_size):
    """
    Calibra estereoscópicamente ambas cámaras y guarda los mapas y parámetros.
    left_data y right_data son tuplas con (camera_matrix, distortion_coeff, roi)
    """
    leftCameraMatrix, leftDistCoeffs, _ = left_data
    rightCameraMatrix, rightDistCoeffs, _ = right_data

    # Convertir los puntos a numpy arrays
    leftImagePoints = np.asarray(imgpointsLeft, dtype=np.float32)
    rightImagePoints = np.asarray(imgpointsRight, dtype=np.float32)
    
    TERMINATION_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 1e-6)
    
    ret, newLeftCameraMatrix, newLeftDistCoeffs, newRightCameraMatrix, newRightDistCoeffs, \
    rotationMatrix, translationVector, essentialMatrix, fundamentalMatrix = cv2.stereoCalibrate(
        objpointsLeft, leftImagePoints, rightImagePoints,
        leftCameraMatrix, leftDistCoeffs,
        rightCameraMatrix, rightDistCoeffs,
        image_size,
        flags=cv2.CALIB_FIX_INTRINSIC + cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_ASPECT_RATIO,
        criteria=TERMINATION_CRITERIA)
    
    print("StereoCalibrate RMS:", ret)
    
    # Rectificación estereoscópica
    R1, R2, P1, P2, Q, leftRoi, rightRoi = cv2.stereoRectify(
        newLeftCameraMatrix, newLeftDistCoeffs,
        newRightCameraMatrix, newRightDistCoeffs,
        image_size, rotationMatrix, translationVector,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    
    leftMapX, leftMapY = cv2.initUndistortRectifyMap(newLeftCameraMatrix, newLeftDistCoeffs, R1, P1, image_size, cv2.CV_16SC2)
    rightMapX, rightMapY = cv2.initUndistortRectifyMap(newRightCameraMatrix, newRightDistCoeffs, R2, P2, image_size, cv2.CV_16SC2)
    
    calib_dir = f'./calibration_data960/{image_size[1]}p'
    stereo_file = os.path.join(calib_dir, 'stereo_camera_calibration.npz')
    
    np.savez_compressed(stereo_file,
                        imageSize=image_size,
                        leftMapX=leftMapX, leftMapY=leftMapY,
                        rightMapX=rightMapX, rightMapY=rightMapY,
                        Q=Q, leftRoi=leftRoi, rightRoi=rightRoi,
                        leftProjection=P1, rightProjection=P2,
                        leftCameraMatrix=newLeftCameraMatrix, leftDistortionCoeff=newLeftDistCoeffs,
                        rightCameraMatrix=newRightCameraMatrix, rightDistortionCoeff=newRightDistCoeffs)
    return stereo_file

# ========================================================
# Funciones para mostrar resultados de undistortion y rectificación
# ========================================================
def show_undistorted_images(image_shape):
    """
    Muestra imágenes undistortadas usando los mapas guardados.
    Se corrige la verificación de claves en el npz usando 'in'.
    """
    DIM = (image_shape[1], image_shape[0])
    calib_dir = f'./calibration_data960/{image_shape[0]}p'
    
    # Cargar datos de la cámara izquierda
    left_file = os.path.join(calib_dir, 'camera_calibration_left.npz')
    try:
        npz_file = np.load(left_file)
        if 'map1' in npz_file.files and 'map2' in npz_file.files:
            map1_left = npz_file['map1']
            map2_left = npz_file['map2']
            roi_left = npz_file['roi']
        else:
            print("Datos de calibración izquierda corruptos.")
            return
    except Exception as e:
        print(f"Datos de calibración izquierda no encontrados en {left_file}: {e}")
        return
    
    # Cargar datos de la cámara derecha
    right_file = os.path.join(calib_dir, 'camera_calibration_right.npz')
    try:
        npz_file = np.load(right_file)
        if 'map1' in npz_file.files and 'map2' in npz_file.files:
            map1_right = npz_file['map1']
            map2_right = npz_file['map2']
            roi_right = npz_file['roi']
        else:
            print("Datos de calibración derecha corruptos.")
            return
    except Exception as e:
        print(f"Datos de calibración derecha no encontrados en {right_file}: {e}")
        return

    # Nota: En una implementación completa se deberían pasar las imágenes a undistort,
    # pero aquí se asume que se usan las últimas imágenes procesadas.
    global gray_small_left, gray_small_right
    undistorted_left = cv2.remap(gray_small_left, map1_left, map2_left, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    x, y, w_roi, h_roi = roi_left
    undistorted_left = undistorted_left[y:y+h_roi, x:x+w_roi]
    
    undistorted_right = cv2.remap(gray_small_right, map1_right, map2_right, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    x, y, w_roi, h_roi = roi_right
    undistorted_right = undistorted_right[y:y+h_roi, x:x+w_roi]
    
    cv2.imshow('Izquierda UNDISTORTED', undistorted_left)
    cv2.imshow('Derecha UNDISTORTED', undistorted_right)
    cv2.waitKey(0)
    if writeUndistortedImages:
        cv2.imwrite("undistorted_left.jpg", undistorted_left)
        cv2.imwrite("undistorted_right.jpg", undistorted_right)

def show_stereo_rectification(image_shape):
    """
    Muestra el resultado de la rectificación estereoscópica.
    Se corrige la verificación de tamaño de imagen y se utiliza f-string.
    """
    DIM = (image_shape[1], image_shape[0])
    calib_dir = f'./calibration_data960/{image_shape[0]}p'
    stereo_file = os.path.join(calib_dir, 'stereo_camera_calibration.npz')
    
    try:
        npz_file = np.load(stereo_file)
    except Exception as e:
        print(f"Datos de calibración estereoscópica no encontrados en {stereo_file}: {e}")
        exit(0)
    
    leftMapX = npz_file['leftMapX']
    leftMapY = npz_file['leftMapY']
    leftRoi = npz_file['leftRoi']
    rightMapX = npz_file['rightMapX']
    rightMapY = npz_file['rightMapY']
    rightRoi = npz_file['rightRoi']
    
    if not os.path.isfile(imageToDisp):
        print(f"No se puede leer la imagen: {imageToDisp}")
        exit(0)
    
    pair_img = cv2.imread(imageToDisp, 0)
    height_img, width_img = pair_img.shape[:2]
    expected_width = photo_width // 2  # Resolución de cada cámara (side-by-side)
    expected_total_width = expected_width * 2
    
    if (width_img != expected_total_width) or (height_img != photo_height):
        # Si es proporcional se redimensiona
        if (width_img / expected_total_width) == (height_img / photo_height):
            pair_img = cv2.resize(pair_img, (expected_total_width, photo_height), interpolation=cv2.INTER_CUBIC)
        else:
            print("Tamaño de imagen incorrecto. Elige una imagen adecuada.")
            exit(0)
    
    imgLTest = pair_img[0:photo_height, 0:expected_width]
    imgRTest = pair_img[0:photo_height, expected_width:]
    
    imgL = cv2.remap(imgLTest, leftMapX, leftMapY, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    imgR = cv2.remap(imgRTest, rightMapX, rightMapY, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    x, y, w_roi, h_roi = leftRoi
    imgL = imgL[y:y+h_roi, x:x+w_roi]
    x, y, w_roi, h_roi = rightRoi
    imgR = imgR[y:y+h_roi, x:x+w_roi]
    
    cv2.imshow('Izquierda STEREO CALIBRATED', imgL)
    cv2.imshow('Derecha STEREO CALIBRATED', imgR)
    cv2.imwrite("rectified_left.jpg", imgL)
    cv2.imwrite("rectified_right.jpg", imgR)
    cv2.waitKey(0)

# ======================
# Función principal (main)
# ======================
def main():
    # Procesar imágenes de calibración
    # La forma de la imagen se obtiene de la última imagen procesada (grayL.shape)
    (objpointsL, imgpointsL, objpointsR, imgpointsR, image_shape) = process_calibration_images()
    
    # Calibrar individualmente cada cámara
    print("Calibración cámara izquierda...")
    left_data = calibrate_one_camera(objpointsL, imgpointsL, 'left', image_shape)
    print("Calibración cámara derecha...")
    right_data = calibrate_one_camera(objpointsR, imgpointsR, 'right', image_shape)
    
    # Calibración estereoscópica
    print("Calibración estereoscópica...")
    stereo_file = calibrate_stereo_cameras(left_data, right_data, (img_width, img_height))
    print("Calibración completa!")
    
    if showSingleCamUndistortionResults:
        show_undistorted_images(image_shape)
    
    if showStereoRectificationResults:
        show_stereo_rectification(image_shape)

if __name__ == "__main__":
    main()
