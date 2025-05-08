import cv2
import numpy as np

# Diccionario de ArUco (ajústalo o agrega otros si es necesario)
ARUCO_DICT = {
    "DICT_4x4_250": cv2.aruco.DICT_4X4_250
}

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

def markCorners(image, markers):
    """
    Dibuja sobre la imagen:
      - Círculos azules en todas las esquinas (con el índice).
      - Un círculo rojo en la esquina seleccionada:
            * Si ID ≤ 79 => esquina inferior izquierda (índice 3)
            * Caso contrario  => esquina superior izquierda (índice 0)
      - El ID del marcador cerca de la esquina seleccionada.
    """
    image_out = image.copy()
    for m in markers:
        marker_id = m['id']
        # Selección de esquina según el ID
        if marker_id <= 79:
            corner_index = 2
          # esquina inferior izquierda
        else:
            corner_index = 1  # esquina superior izquierda

        selected_corner = m['corners'][corner_index]

        # Dibujar la esquina seleccionada (círculo rojo)
        cv2.circle(image_out, tuple(selected_corner.astype(int)), 6, (0, 0, 255), -1)

        # Dibujar todas las esquinas con círculos azules y numerarlas para referencia
        for j, corner in enumerate(m['corners']):
            cv2.circle(image_out, tuple(corner.astype(int)), 4, (255, 0, 0), -1)
            cv2.putText(image_out, f"{j}", tuple(corner.astype(int) + np.array([3, -3])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Dibujar el ID del marcador cerca de la esquina seleccionada
        cv2.putText(image_out, f"ID: {marker_id}", tuple(selected_corner.astype(int) + np.array([10, 10])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    return image_out

def main():
    # Rutas de las imágenes (ajústalas a tus archivos)
    imgL_path = "imgL.png"
    imgR_path = "imgR.png"
    
    # Cargar imágenes
    imgL = cv2.imread(imgL_path)
    imgR = cv2.imread(imgR_path)
    if imgL is None or imgR is None:
        print("Error: No se pudieron cargar las imágenes.")
        return

    # Convertir a escala de grises para la detección de marcadores
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    # Detectar marcadores en cada imagen
    markersL = detectMarkers(grayL)
    markersR = detectMarkers(grayR)

    # Marcar las esquinas seleccionadas y todas las esquinas en cada imagen
    imgL_marked = markCorners(imgL, markersL)
    imgR_marked = markCorners(imgR, markersR)

    # Mostrar resultados
    cv2.imshow("Imagen Izquierda - Esquinas", imgL_marked)
    cv2.imshow("Imagen Derecha - Esquinas", imgR_marked)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
