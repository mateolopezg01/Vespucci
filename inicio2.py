#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────
#  inicializacion.py
#
#  • Carga la calibración estéreo (mapas, proyecciones y distorsiones)
#  • Rectifica la primera pareja (imgL, imgR)
#  • Recalcula [R|t] → P1,P2 con solvePnP + esquinas ArUco sub‑píxel
#  • Calcula afinidad (T,b) con la tabla W de pnp.py
#  • En terminal:
#      – IDs de marcadores usados
#      – reproj. 2D (μ/max en px) por cámara
#      – alineación 3D (μ/max en mm)
#  • Devuelve: T, b, P1, P2, origin, maps, distL, distR
# ──────────────────────────────────────────────────────────────
import cv2, numpy as np

# ——— ArUco + sub‑píxel ———
ARUCO_ID = cv2.aruco.DICT_4X4_250
CRIT     = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4)

def detect_subpix(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    det  = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(ARUCO_ID),
        cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(gray)
    mk=[]
    if ids is not None:
        for c,i in zip(corners,ids.flatten()):
            cv2.cornerSubPix(gray, c, (5,5), (-1,-1), CRIT)
            mk.append({"id":int(i), "corners":c.reshape(4,2)})
    return mk

# ——— tabla W + offsets (igual que pnp.py) ———
def build_W0():
    A4_w, sep_x, m, line, back = 210,200,10,40,27
    W0 = np.zeros((120,3),np.float32)
    for n in range(120):
        if n<80:
            W0[n] = [back+(n//8)*60, 2+line+m+(n%8)*50, 0]
        elif n<100:
            W0[n] = [m+50*((n-80)%4), 0, line+m+60*((n-80)//4)]
        else:
            W0[n] = [A4_w+sep_x+m+50*((n-100)%4), 0, line+m+60*((n-100)//4)]
    return W0

OFF = np.zeros((2,4,3),np.float32)
OFF[0,1]=[0,40,0]; OFF[0,2]=[40,40,0]; OFF[0,3]=[40,0,0]
OFF[1,1]=[40,0,0]; OFF[1,2]=[40,0,40]; OFF[1,3]=[0,0,40]
def world_corner(mid,idx,W0):
    return W0[mid] + OFF[0 if mid<80 else 1, idx]

# ——— recalc_P: solvePnP + reproyección px ———
def recalc_P(imgL, imgR, P1r, P2r, distL, distR, W0):
    K1, K2 = P1r[:,:3], P2r[:,:3]
    mkL, mkR = detect_subpix(imgL), detect_subpix(imgR)
    ids = sorted({m["id"] for m in mkL} & {m["id"] for m in mkR} & set(range(120)))
    if not ids:
        raise RuntimeError("No hay marcadores comunes")

    obj, ptsL, ptsR = [], [], []
    for i in ids:
        mL = next(m for m in mkL if m["id"]==i)
        mR = next(m for m in mkR if m["id"]==i)
        for k in range(4):
            obj.append(world_corner(i,k,W0))
            ptsL.append(mL["corners"][k])
            ptsR.append(mR["corners"][k])

    obj = np.asarray(obj, dtype=np.float32)
    ptsL= np.asarray(ptsL,dtype=np.float32)
    ptsR= np.asarray(ptsR,dtype=np.float32)

    flag = cv2.SOLVEPNP_IPPE_SQUARE if len(obj)==4 else cv2.SOLVEPNP_ITERATIVE
    okL, rL, tL = cv2.solvePnP(obj, ptsL, K1, distL, flags=flag)
    okR, rR, tR = cv2.solvePnP(obj, ptsR, K2, distR, flags=flag)
    if not (okL and okR):
        raise RuntimeError("solvePnP falló")

    RL,_ = cv2.Rodrigues(rL)
    RR,_ = cv2.Rodrigues(rR)
    P1 = K1 @ np.hstack([RL, tL])
    P2 = K2 @ np.hstack([RR, tR])

    # reproyección
    projL,_ = cv2.projectPoints(obj, rL, tL, K1, distL)
    projR,_ = cv2.projectPoints(obj, rR, tR, K2, distR)
    errL = np.linalg.norm(projL.reshape(-1,2) - ptsL, axis=1)
    errR = np.linalg.norm(projR.reshape(-1,2) - ptsR, axis=1)

    print("❖ solvePnP:")
    print(f"  IDs usados ({len(ids)}): {ids}")
    print(f"  reproj LEFT   μ={errL.mean():.2f}px  max={errL.max():.2f}px")
    print(f"  reproj RIGHT  μ={errR.mean():.2f}px  max={errR.max():.2f}px")

    return P1, P2, ids

# ——— obtTyb (igual que antes) ———
def obtTyb(pRefs, W):
    n = len(W)
    A = np.zeros((3*n,12), np.float32)
    y = np.zeros((3*n,1),  np.float32)
    for k in range(n):
        X = pRefs[k]
        for l in range(3):
            A[3*k+l, 4*l:4*l+3] = X
            A[3*k+l, 4*l+3]   = 1
            y[3*k+l]          = W[k,l]
    x, *_ = np.linalg.lstsq(A, y, rcond=None)
    T = x.reshape(3,4)[:,:3]
    b = x.reshape(3,4)[:,3]
    return T, b

# ——— TyB_aruco: triangulación + error 3D mm ———
def TyB_aruco(imgL, imgR, P1, P2, W0, ids):
    p3d, Wpts = [], []
    for i in ids:
        mL = next(m for m in detect_subpix(imgL) if m["id"]==i)
        mR = next(m for m in detect_subpix(imgR) if m["id"]==i)
        cL = np.mean(mL["corners"],axis=0).reshape(2,1).astype(np.float32)
        cR = np.mean(mR["corners"],axis=0).reshape(2,1).astype(np.float32)
        X4 = cv2.triangulatePoints(P1,P2,cL,cR)
        X3 = (X4[:3]/X4[3]).ravel()
        if i>79: 
            print(i," ",X3)
        p3d.append(X3)
        Wpts.append(world_corner(i,0,W0))

    pArr = np.asarray(p3d, dtype=np.float32)
    WArr = np.asarray(Wpts, dtype=np.float32)
    T, b = obtTyb(pArr, WArr)

    pred = pArr @ T.T + b
    regresores = entrenar_regresiones(pred, WArr)

    err3 = np.linalg.norm(pred - WArr, axis=1)
    print("❖ alineación 3D:")
    print(f"  marcadores ({len(ids)}): {ids}")
    print(f"  error 3D    μ={err3.mean():.2f}mm  max={err3.max():.2f}mm")
    pred_corr = np.vstack([reg.predict(pred) for reg in regresores]).T
    err3_corr = np.linalg.norm(pred_corr - WArr, axis=1)
    print(f"  error 3D  corregido  μ={err3_corr.mean():.2f}mm  max={err3_corr.max():.2f}mm")

    # origen para graficado
    origin = W0[87] + np.array([105,35,0], dtype=np.float32)
    print(origin)
    return T, b, origin, regresores

from sklearn.linear_model import LinearRegression

def entrenar_regresiones(pRefs, W):
    regresores = []
    for i in range(3):  # para x, y, z
        reg = LinearRegression()
        reg.fit(pRefs, W[:, i])
        regresores.append(reg)
    return regresores


# ——— PÚBLICO ———
def inicializar(imgL, imgR):
    cal = np.load("./calibration_data960/960p/stereo_camera_calibration.npz")
    maps   = (cal["leftMapX"],cal["leftMapY"],cal["rightMapX"],cal["rightMapY"])
    distL, distR = cal["leftDistortionCoeff"], cal["rightDistortionCoeff"]
    P1r, P2r     = cal["leftProjection"],      cal["rightProjection"]

    # rectificación ligera
    L = cv2.remap(imgL, maps[0], maps[1], cv2.INTER_LINEAR)
    R = cv2.remap(imgR, maps[2], maps[3], cv2.INTER_LINEAR)


    W0 = build_W0()
    P1, P2, ids = recalc_P(L, R, P1r, P2r, distL, distR, W0)
    T, b, origin, regresores = TyB_aruco(L, R, P1, P2, W0, ids)

    # ← **no cambia la firma de retorno**
    return T, b, P1, P2, origin, maps, distL, distR, regresores
