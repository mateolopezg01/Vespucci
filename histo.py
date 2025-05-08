import cv2
import os
import numpy as np

def extraer_y_recortar_frames_similarity(ruta_video, carpeta_salida, x, y, w, h, umbral_porcentaje=0.99, umbral_diff=10):
    """
    Extrae los frames de un video, recorta cada frame y los guarda en 'carpeta_salida'
    solo si no son al menos un 99% similares al frame anterior basado en diferencias de píxeles.

    Parámetros:
    ruta_video (str): Ruta al archivo de video.
    carpeta_salida (str): Carpeta donde se guardarán los frames recortados.
    x, y (int): Coordenadas de la esquina superior izquierda del recorte.
    w, h (int): Ancho y alto del recorte.
    umbral_porcentaje (float): Umbral de porcentaje para decidir si guardar el frame.
                                 Por defecto, 0.99 (99%).
    umbral_diff (int): Umbral de diferencia por canal para considerar píxeles similares.
                       Por defecto, 10.
    """

    # Crear carpeta de salida si no existe
    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)
    
    # Iniciar captura de video
    cap = cv2.VideoCapture(ruta_video)

    # Verificar si el video se abrió correctamente
    if not cap.isOpened():
        print(f"No se pudo abrir el archivo de video: {ruta_video}")
        return

    contador_frames = 0
    ultimo_frame_recortado = None  # Para almacenar el frame anterior

    while True:
        ret, frame = cap.read()
        if not ret:
            # No se pudo leer el siguiente frame o se acabó el video
            break
        
        # Recortar el frame: [y : y+h, x : x+w]
        frame_recortado = frame[y:y+h, x:x+w]

        # Si no es el primer frame, comparar con el anterior
        if ultimo_frame_recortado is not None:
            # Calcular la diferencia absoluta entre los frames
            diferencia = cv2.absdiff(frame_recortado, ultimo_frame_recortado)
            
            # Convertir la diferencia a escala de grises
            diferencia_gray = cv2.cvtColor(diferencia, cv2.COLOR_BGR2GRAY)
            
            # Aplicar un umbral para considerar diferencias significativas
            _, diferencia_thresh = cv2.threshold(diferencia_gray, umbral_diff, 255, cv2.THRESH_BINARY)
            
            # Contar el número de píxeles que son diferentes
            num_píxeles_diferentes = cv2.countNonZero(diferencia_thresh)
            total_píxeles = diferencia_thresh.size
            
            # Calcular el porcentaje de píxeles similares
            porcentaje_similares = 1 - (num_píxeles_diferentes / total_píxeles)
            
            if porcentaje_similares >= umbral_porcentaje:
                # Los frames son suficientemente similares, no guardar
                print(f"Frame {contador_frames} omitido por alta similitud ({porcentaje_similares*100:.2f}% similares).")
                continue  # Saltar a la siguiente iteración sin guardar
        
        # Guardar el frame recortado como imagen
        nombre_imagen = f"frame_{contador_frames:05d}.png"
        ruta_imagen = os.path.join(carpeta_salida, nombre_imagen)
        cv2.imwrite(ruta_imagen, frame_recortado)

        # Actualizar el frame anterior
        ultimo_frame_recortado = frame_recortado.copy()
        contador_frames += 1

    cap.release()
    print("Proceso completado. Frames extraídos (y recortados) cuando fueron significativamente diferentes al anterior.")

# Ejemplo de uso:
if __name__ == "__main__":
    # Ajusta estos parámetros según tus necesidades
    ruta_del_video = f"C:/Users/anaib/Downloads/Copia_de_Opcion_logo_1.mp4" # Reemplaza con la ruta de tu video
    carpeta_de_salida = "frames_recortados"      # Carpeta donde se guardarán las imágenes
    x, y, w, h = 0, 0, 1280, 1080               # Coordenadas y tamaño del recorte
    umbral_porcentaje = 0.98                       # 99%
    umbral_diff = 10                                # Diferencia máxima por canal

    extraer_y_recortar_frames_similarity(
        ruta_del_video,
        carpeta_de_salida,
        x, y, w, h,
        umbral_porcentaje,
        umbral_diff
    )


