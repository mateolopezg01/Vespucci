import cv2
import numpy as np
from picamera import PiCamera
from time import sleep

def load_calibration(img_height):
    """Carga los datos de calibración de la cámara."""
    try:           
        npz_file = np.load('./calibration_data/{}p/stereo_camera_calibration.npz'.format(img_height))
        if all(key in npz_file.files for key in ['leftMapX', 'leftMapY', 'rightMapX', 'rightMapY', 'leftProjection', 'rightProjection']):
            print("Camera calibration data has been found in cache.")
            mapxL = npz_file['leftMapX']
            mapyL = npz_file['leftMapY']
            mapxR = npz_file['rightMapX']
            mapyR = npz_file['rightMapY']
            P1 = npz_file['leftProjection']
            P2 = npz_file['rightProjection']
            lmtx = npz_file['leftCameraMatrix']
            ldst = npz_file['leftDistortionCoeff']
            return mapxL, mapyL, mapxR, mapyR, P1, P2
        else:
            print("Camera data file found but data corrupted.")
            exit(0)
    except Exception as e:
        print(f"Error loading calibration data: {e}")
        exit(0)

def capture_images():
    """Captura imágenes de la StereoPi."""
    camera = PiCamera(stereo_mode='side-by-side', resolution=(1280, 480))  # Configuración para StereoPi
    camera.start_preview()
    print("Preparando las cámaras...")

    sleep(2)  # Tiempo para estabilizar las cámaras
    camera.capture("stereo_image.jpg")  # Captura la imagen combinada
    camera.stop_preview()
    camera.close()

    # Divide la imagen combinada en izquierda y derecha
    stereo_image = cv2.imread("stereo_image.jpg")
    if stereo_image is None:
        print("Error al capturar la imagen.")
        return None, None

    imgL = stereo_image[:, :640, :]  # Imagen izquierda
    imgR = stereo_image[:, 640:, :]  # Imagen derecha
    return imgL, imgR

def process_images(imgL, imgR, mapxL, mapyL, mapxR, mapyR):
    """Aplica la remapeo a las imágenes usando los datos de calibración."""
    imgL_rectified = cv2.remap(imgL, mapxL, mapyL, cv2.INTER_LINEAR)
    imgR_rectified = cv2.remap(imgR, mapxR, mapyR, cv2.INTER_LINEAR)
    return imgL_rectified, imgR_rectified

def refPoints(imgL, imgR, P1, P2):
    bboxL = cv2.selectROI("Imagen izquierda, seleccione nuevo punto de referencia", imgL, False, False)
    centerL = np.array([(bboxL[0] + bboxL[0] + bboxL[2]) / 2, (bboxL[1] + bboxL[1] + bboxL[3]) / 2], dtype=np.float32)
    cv2.destroyAllWindows()
    
    bboxR = cv2.selectROI("Imagen derecha, seleccione mismo punto de referencia", imgR, False, False)
    centerR = np.array([(bboxR[0] + bboxR[0] + bboxR[2]) / 2, (bboxR[1] + bboxR[1] + bboxR[3]) / 2], dtype=np.float32)
    cv2.destroyAllWindows()

    centerL_hom = np.array([centerL[0], centerL[1], 1.0]).reshape(3, 1)  # Coordenadas homogéneas
    centerR_hom = np.array([centerR[0], centerR[1], 1.0]).reshape(3, 1)

    points_4D_hom = cv2.triangulatePoints(P1, P2, centerL_hom, centerR_hom)
    if points_4D_hom[3] != 0:
        points_3D = points_4D_hom[:3] / points_4D_hom[3]
        points_3D = points_3D.flatten()
    else:
        print("Error en la triangulación: divisor es cero.")
        points_3D = np.array([0, 0, 0])

    return points_3D

# Función principal
def main():
    img_height = 480  # Ajusta según la resolución de la calibración
    mapxL, mapyL, mapxR, mapyR, P1, P2 = load_calibration(img_height)
    
    imgL, imgR = capture_images()
    if imgL is None or imgR is None:
        print("Error al capturar imágenes. Asegúrate de que la StereoPi esté configurada correctamente.")
        return
    
    imgL_rectified, imgR_rectified = process_images(imgL, imgR, mapxL, mapyL, mapxR, mapyR)
    
    puntos_3D = []
    print("Selecciona los puntos de referencia. Presiona 'Escape' para terminar.")
    
    while True:
        try:
            punto_3D = refPoints(imgL_rectified, imgR_rectified, P1, P2)
            puntos_3D.append(punto_3D)
            print(f"Punto guardado: {punto_3D}")
        except Exception as e:
            print("Selección cancelada o error:", str(e))
            break

        continuar = input("¿Deseas seleccionar otro punto? (s/n): ")
        if continuar.lower() != 's':
            break
    
    puntos_3D = np.array(puntos_3D)
    np.savez("puntos_referencia.npz", puntos=puntos_3D)
    print(f"Puntos guardados en 'puntos_referencia.npz':\n{puntos_3D}")

if __name__ == "__main__":
    main()
