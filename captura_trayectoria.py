#!/usr/bin/env python3
# ------------------------------------------------------------
# Captura de ext2  ―  se graba en mm después de presionar <SPACE>
# ------------------------------------------------------------
import numpy as np, cv2, threading, time
from segmentardo import extremos_segmento
from graficado2  import graficado2
from inicio2     import inicializar

# ---------- Parámetros de video ----------
STREAM_URL        = 'udp://169.254.144.155:3000'
VIDEO_W, VIDEO_H  = 1280, 960
PUNTOS            = 4
ALPHA             = 0.1          # suavizado
UMBRAL_BIN        = 250          # para LEDs
KERNEL            = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

# ---------- Captura en hilo ----------
class ThreadedVideoCapture:
    def __init__(self,url):
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not self.cap.isOpened(): raise RuntimeError("Stream no disponible")
        self.lock, self.stopped, self.frame = threading.Lock(), False, None
        threading.Thread(target=self.update,daemon=True).start()
    def update(self):
        while not self.stopped:
            ok,frm = self.cap.read()
            if ok:
                with self.lock: self.frame=frm.copy()
            else: time.sleep(0.01)
    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()
    def stop(self):
        self.stopped=True; self.cap.release()

# ---------- Utilidades ----------
def centros(img):
    g   = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _,b = cv2.threshold(g,UMBRAL_BIN,255,cv2.THRESH_BINARY)
    b   = cv2.morphologyEx(b,cv2.MORPH_OPEN,KERNEL)
    cnt,_ = cv2.findContours(b,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    c=[]
    for ct in cnt:
        m=cv2.moments(ct)
        if m["m00"]:
            c.append((int(m["m10"]/m["m00"]), int(m["m01"]/m["m00"])))
    return c

def punto3d(cL,cR,P1,P2,T,b):                  # devuelve cm
    cL,cR = [np.array(p,dtype=np.float32).reshape(2,1) for p in (cL,cR)]
    X4    = cv2.triangulatePoints(P1,P2,cL,cR)
    X3    = (X4[:3]/X4[3]).ravel()
    return T@X3 + b

# ---------- Inicio ----------
cap = ThreadedVideoCapture(STREAM_URL)

# frame para inicializar
frm0 = None
while frm0 is None: frm0=cap.read()
L0,R0 = frm0[:VIDEO_H,:VIDEO_W//2], frm0[:VIDEO_H,VIDEO_W//2:]
T,b,P1,P2,origin,(mxL,myL,mxR,myR),_,_,_ = inicializar(L0,R0)

coords          = [[] for _ in range(PUNTOS)]
traj_ext2_mm    = []
grabando        = False

# Bucle de reintentos para obtener el primer frame
max_attempts = 10
attempts = 0
initial_frame = None
while attempts < max_attempts and initial_frame is None:
    initial_frame = cap.read()
    if initial_frame is None:
        print(f"Intento {attempts+1}: No se pudo capturar el frame inicial, reintentando...")
        time.sleep(0.1)
        attempts += 1
print("▶ Stream iniciado ― <i>SPACE</i> empieza a grabar, <i>q</i> termina.")
while True:
    fr = cap.read()
    if fr is None: continue
    L = cv2.remap(fr[:VIDEO_H,:VIDEO_W//2], mxL,myL, cv2.INTER_LINEAR)
    R = cv2.remap(fr[:VIDEO_H,VIDEO_W//2:],  mxR,myR, cv2.INTER_LINEAR)

    cL = sorted(centros(L),key=lambda p:p[0])
    cR = sorted(centros(R),key=lambda p:p[0])

    if min(len(cL),len(cR))>=PUNTOS:
        newc=[None]*PUNTOS
        for i in range(PUNTOS):
            p = punto3d(cL[i],cR[i],P1,P2,T,b)  
            if all(isinstance(c, np.ndarray) for c in coords):
                prev=np.vstack(coords)
                idx=np.argmin(np.linalg.norm(prev-p,axis=1))
                p   = ALPHA*p + (1-ALPHA)*prev[idx]
                newc[idx]=p
            else: newc[i]=p
        coords=newc                                # ← actualizamos

        # --- Llamar a extremos_segmento SOLO si los 4 puntos son válidos ---
        if all(isinstance(c, np.ndarray) for c in coords):
            ext1, ext2 = extremos_segmento(coords)       # ahora sí seguro
            if grabando:
                traj_ext2_mm.append(ext2)           # cm → mm
            graficado2(coords)                           # overlay habitual
        # Si falta algún punto todavía, esperamos al siguiente frame


    key=cv2.waitKey(1)&0xFF
    if key==ord('q'): break
    if key==ord(' '): 
        grabando=True
        print("⏺️  Grabación de ext2 iniciada…")

cap.stop(); cv2.destroyAllWindows()

traj_ext2_mm=np.asarray(traj_ext2_mm)
np.save('trayectoria_ext2_mm.npy', traj_ext2_mm)
np.savetxt('trayectoria_ext2_mm.csv', traj_ext2_mm, delimiter=',')

print(f"✔ Se guardaron {len(traj_ext2_mm)} puntos en trayectoria_ext2_mm.*")
