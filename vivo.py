import cv2
import time
import numpy as np
stream_url = 'udp://169.254.144.155:3000'  # Replace with your actual stream URL
cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
#cap.set(cv2.CAP_PROP_BUFFERSIZE,2)
if not cap.isOpened():
    print("Error: Unable to open video stream")
    exit()
print("Esperando primer frame válido...")
for _ in range(100):
	ret, frame = cap.read()
	if ret and frame is not None:
		print("Primer frame recibido")
		break
	time.sleep(0.1)  # pequeña pausa entre intentos
lower = np.array([250, 250, 250])    # Rojo mínimo
upper = np.array([255, 255, 255])  # Rojo máximo
(width, height)=(5,5)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame")
        break
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY)
    eroded_img = cv2.erode(binary_img, kernel, iterations=1)

    # Aplica la dilatación para rellenar huecos pequeños
    processed_img = cv2.dilate(eroded_img, kernel, iterations=1)   
    
    # Crear máscara binaria: píxeles dentro del rango son blancos (255), los demás negros (0)
    mask = cv2.inRange(frame, lower, upper)

    # Aplicar la máscara a la imagen original
    result = cv2.bitwise_and(frame, frame, mask=mask)
    # Display the frame
    cv2.imshow('StereoPi Stream', frame)
    # cv2.imshow('StereoPi Stream', frame)
    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
