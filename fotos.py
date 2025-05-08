import cv2
import time
import numpy as np
import os

stream_url = 'udp://169.254.144.155:3000'  # Reemplaza con tu URL real
cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("Error: Unable to open video stream")
    exit()

print("Esperando primer frame válido...")
for _ in range(100):
    ret, frame = cap.read()
    if ret and frame is not None:
        print("Primer frame recibido")
        break
    time.sleep(0.1)  # Pequeña pausa entre intentos

video_height = 960
video_width = 1280

# Crea la carpeta "pairs960" si no existe
if not os.path.exists("pairs960"):
    os.makedirs("pairs960")

def captura(cap):
    ret, frame = cap.read()
    if not ret:
        print("No se recibió frame de la cámara. Saliendo...")
        return None, None, None
    else:
        # Separa la imagen en dos partes (izquierda y derecha)
        imgL = frame[:video_height, 0:video_width//2, :]
        imgR = frame[:video_height, video_width//2:video_width, :]
    return frame, imgL, imgR

photo_counter = 0

while True:
    frame, imgL, imgR = captura(cap)
    if frame is None:
        break
    
    cv2.imshow('StereoPi Stream', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):  # Al presionar la barra espaciadora
        photo_counter += 1
        # Si el contador es menor a 10, agrega un cero delante del número
        if photo_counter < 10:
            # filenameL = f"pairs960/left_0{photo_counter}.png"
            # filenameR = f"pairs960/right_0{photo_counter}.png"
            filenameL = f"imgL3.png"
            filenameR = f"imgR3.png"
        else:
            filenameL = f"pairs960/left_{photo_counter}.png"
            filenameR = f"pairs960/right_{photo_counter}.png"
        cv2.imwrite(filenameL, imgL)
        cv2.imwrite(filenameR, imgR)
        print(f"Imagen guardada: {filenameL} y {filenameR}")

# Libera la captura y cierra las ventanas
cap.release()
cv2.destroyAllWindows()
