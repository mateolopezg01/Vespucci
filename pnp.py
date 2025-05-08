# """
# Comparación de dos métodos para obtener coordenadas 3‑D de
# las esquinas de marcadores ArUco en una StereoPi:

# 1)  Triangulación puntual (cv2.triangulatePoints)
# 2)  Disparidad + matriz Q (StereoSGBM → cv2.reprojectImageTo3D)

# Requiere:
#     • imgL.png / imgR.png                (par rectificado o no)
#     • calibration_data960/...npz         (maps + P1_rect / P2_rect)
# """

# # -----------------------------------------------------------
# #  IMPORTS
# # -----------------------------------------------------------
# import cv2, sys, os
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D    # noqa: F401
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression

# # -----------------------------------------------------------
# #  PARÁMETROS GLOBALES
# # -----------------------------------------------------------
# BASELINE_MM = 110.0          # distancia entre cámaras   (mm)
# NUM_DISP    = 128            # múltiplo de 16
# BLKSZ       = 5              # blockSize SGBM

# # -----------------------------------------------------------
# #  ARUCO DICT
# # -----------------------------------------------------------
# ARUCO_DICT = {
#     "DICT_4x4_250": cv2.aruco.DICT_4X4_250,
# }

# def detectMarkers(img, arucoDict="DICT_4x4_250"):
#     dic  = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[arucoDict])
#     det  = cv2.aruco.ArucoDetector(dic, cv2.aruco.DetectorParameters())
#     corners, ids, _ = det.detectMarkers(img)
#     markers = []
#     if ids is not None:
#         for c, i in zip(corners, ids.flatten()):
#             markers.append({"id": int(i), "corners": c.reshape(4, 2)})
#     return markers

# # -----------------------------------------------------------
# #  TABLA W – esquina 0 de cada marcador
# # -----------------------------------------------------------
# def build_W0():
#     A4_w, sep_x, margin, line, back_margin = 210, 200, 10, 40, 27
#     W0 = np.zeros((120, 3), dtype=np.float32)
#     for n in range(120):
#         if n < 80:
#             W0[n] = [back_margin + (n // 8) * 60,
#                      2 + line + margin + (n % 8) * 50,
#                      0]
#         elif n < 100:
#             W0[n] = [margin + 50 * ((n - 80) % 4),
#                      0,
#                      line + margin + 60 * ((n - 80) // 4)]
#         else:
#             W0[n] = [A4_w + sep_x + margin + 50 * ((n - 100) % 4),
#                      0,
#                      line + margin + 60 * ((n - 100) // 4 )]
#     return W0

# OFFSETS = np.zeros((2, 4, 3), dtype=np.float32)
# OFFSETS[0, 1] = [0, 40, 0];  OFFSETS[0, 2] = [40, 40, 0]; OFFSETS[0, 3] = [40, 0, 0]
# OFFSETS[1, 1] = [40, 0, 0];  OFFSETS[1, 2] = [40, 0, 40]; OFFSETS[1, 3] = [0, 0, 40]

# def world_corner(id_marker: int, idx_corner: int, W0):
#     base = W0[id_marker]
#     grp  = 0 if id_marker < 80 else 1
#     return base + OFFSETS[grp, idx_corner]

# # -----------------------------------------------------------
# #  RE‑ESTIMA P1 / P2 con solvePnP
# # -----------------------------------------------------------
# def recalc_P_using_solvePnP(imgL, imgR, P1_rect, P2_rect, W0, distL, distR):
#     K1, K2 = P1_rect[:, :3], P2_rect[:, :3]
#     mkL, mkR = detectMarkers(imgL), detectMarkers(imgR)
#     ids_common = sorted({m["id"] for m in mkL} & {m["id"] for m in mkR})
#     ids_common = [i for i in ids_common if i < 120]
#     if len(ids_common) == 0:
#         raise RuntimeError("No hay marcadores comunes")

#     objPts, imgL_pts, imgR_pts = [], [], []
#     for idm in ids_common:
#         mL = next(m for m in mkL if m["id"] == idm)
#         mR = next(m for m in mkR if m["id"] == idm)
#         for idx in range(4):
#             objPts.append(world_corner(idm, idx, W0))
#             imgL_pts.append(mL["corners"][idx])
#             imgR_pts.append(mR["corners"][idx])

#     objPts   = np.asarray(objPts,  dtype=np.float32)
#     imgL_pts = np.asarray(imgL_pts, dtype=np.float32)
#     imgR_pts = np.asarray(imgR_pts, dtype=np.float32)

#     okL, rvecL, tvecL = cv2.solvePnP(objPts, imgL_pts, K1, distL)
#     okR, rvecR, tvecR = cv2.solvePnP(objPts, imgR_pts, K2, distR)
#     if not (okL and okR):
#         raise RuntimeError("solvePnP falló")

#     RL, _ = cv2.Rodrigues(rvecL)
#     RR, _ = cv2.Rodrigues(rvecR)
#     P1 = (K1 @ np.hstack((RL, tvecL))).astype(np.float32)
#     P2 = (K2 @ np.hstack((RR, tvecR))).astype(np.float32)

#     print(f"[solvePnP] IDs usados: {ids_common}  —  puntos: {len(objPts)}")
#     return P1, P2

# # -----------------------------------------------------------
# #  TRIANGULACIÓN puntual
# # -----------------------------------------------------------
# def triangulate_all(imgL, imgR, P1, P2, W0):
#     mkL, mkR = detectMarkers(imgL), detectMarkers(imgR)
#     ids_common = sorted({m["id"] for m in mkL} & {m["id"] for m in mkR})
#     ids_common = [i for i in ids_common if i > 0 and i<120]

#     pRefs, W_pts = [], []
#     for idm in ids_common:
#         mL = next(m for m in mkL if m["id"] == idm)
#         mR = next(m for m in mkR if m["id"] == idm)
#         for idx in range(4):
#             cL = mL["corners"][idx].reshape(2, 1)
#             cR = mR["corners"][idx].reshape(2, 1)
#             X4 = cv2.triangulatePoints(P1, P2, cL, cR)
#             pRefs.append((X4[:3] / X4[3]).flatten())
#             W_pts.append(world_corner(idm, idx, W0))
#     return np.asarray(pRefs, dtype=np.float32), np.asarray(W_pts, dtype=np.float32)

# # -----------------------------------------------------------
# #  DISPARIDAD + matriz Q
# # -----------------------------------------------------------
# def build_Q_from_projections(P1_rect, P2_rect, baseline_mm=BASELINE_MM):
#     fx = P1_rect[0, 0]; fy = P1_rect[1, 1]
#     cx = P1_rect[0, 2]; cy = P1_rect[1, 2]
#     cx2 = P2_rect[0, 2]
#     Q = np.array([[1, 0, 0, -cx],
#                   [0, 1, 0, -cy],
#                   [0, 0, 0,  fx],
#                   [0, 0, -1.0 / baseline_mm, (cx - cx2) / baseline_mm]],
#                  dtype=np.float32)
#     return Q

# def disparity_map(imgL, imgR, num_disp=NUM_DISP, block_size=BLKSZ):
#     grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
#     grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
#     stereo = cv2.StereoSGBM_create(
#         minDisparity=0,
#         numDisparities=num_disp,
#         blockSize=block_size,
#         P1=8 * 3 * block_size ** 2,
#         P2=32 * 3 * block_size ** 2,
#         mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
#     disp = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
#     disp[disp <= 0] = np.nan    # evita reproyección inválida
#     return disp

# def sample_points3D_Q(imgL, disp, Q, W0):
#     pts3D = cv2.reprojectImageTo3D(disp, Q)  # (H,W,3)
#     mk = detectMarkers(imgL)
#     pRefs, W_pts = [], []
#     for m in mk:
#         if m["id"] >= 120:
#             continue
#         for idx in range(4):
#             u, v = m["corners"][idx]
#             x, y = int(round(u)), int(round(v))
#             if (0 <= y < pts3D.shape[0]) and (0 <= x < pts3D.shape[1]):
#                 X, Y, Z = pts3D[y, x]
#                 if not np.isfinite(Z):        # disparidad inválida
#                     continue
#                 pRefs.append([X, Y, Z])
#                 W_pts.append(world_corner(m["id"], idx, W0))
#     return np.asarray(pRefs, dtype=np.float32), np.asarray(W_pts, dtype=np.float32)

# # -----------------------------------------------------------
# #  AFINIDAD  y  CORRECCIÓN SISTEMÁTICA
# # -----------------------------------------------------------
# def obtTyb(pRefs, W):
#     n = pRefs.shape[0]
#     A = np.zeros((3*n, 12), dtype=np.float32)
#     y = np.zeros((3*n, 1), dtype=np.float32)
#     for k in range(n):
#         X = pRefs[k]
#         for l in range(3):
#             A[3*k+l, 4*l:4*l+3] = X
#             A[3*k+l, 4*l+3]     = 1
#             y[3*k+l]            = W[k, l]
#     x, *_ = np.linalg.lstsq(A, y, rcond=None)
#     T = x.reshape(3, 4)[:, :3]
#     b = x.reshape(3, 4)[:, 3]
#     return T.astype(np.float32), b.astype(np.float32)

# def systematic_correction(pRefs, W):
#     resid = W - pRefs
#     XY = pRefs[:, :2]
#     poly = PolynomialFeatures(2, include_bias=True)
#     XY2 = poly.fit_transform(XY)
#     corrected = np.empty_like(pRefs)
#     for i in range(3):
#         lr = LinearRegression().fit(XY2, resid[:, i])
#         corrected[:, i] = pRefs[:, i] + lr.predict(XY2)
#     return corrected

# # -----------------------------------------------------------
# #  GRÁFICA
# # -----------------------------------------------------------
# def plot_3d(A, B, title):
#     fig = plt.figure()
#     ax  = fig.add_subplot(111, projection='3d')
#     ax.scatter(*A.T, c='blue', label='Pred')
#     ax.scatter(*B.T, c='red', marker='^', label='GT')
#     ax.set_title(title); ax.legend()
#     plt.show()

# # -----------------------------------------------------------
# #  MAIN
# # -----------------------------------------------------------
# def main():
#     if not os.path.exists("imgL.png") or not os.path.exists("imgR.png"):
#         sys.exit("No se encuentran imgL.png / imgR.png")

#     imgL = cv2.imread("imgL.png")
#     imgR = cv2.imread("imgR.png")

#     npz = np.load("./calibration_data960/960p/stereo_camera_calibration.npz")
#     mapxL, mapyL   = npz["leftMapX"],   npz["leftMapY"]
#     mapxR, mapyR   = npz["rightMapX"],  npz["rightMapY"]
#     P1_rect, P2_rect = npz["leftProjection"], npz["rightProjection"]
#     distL, distR = npz["leftDistortionCoeff"],npz["rightDistortionCoeff"]  # o como lo guardaste

#     # Rectificación
#     imgLr = cv2.remap(imgL, mapxL, mapyL, cv2.INTER_CUBIC)
#     imgRr = cv2.remap(imgR, mapxR, mapyR, cv2.INTER_CUBIC)

#     cv2.imshow("L",imgLr)
#     cv2.imshow("R",imgRr)
#     W0 = build_W0()

#     # Proyecciones refinadas (solvePnP)
#     try:
#         P1, P2 = recalc_P_using_solvePnP(imgLr, imgRr, P1_rect, P2_rect, W0,distL,distR)
#     except RuntimeError as e:
#         print(e); P1, P2 = P1_rect, P2_rect

#     # ----------  MÉTODO A : TRIANGULATEPOINTS ----------
#     pTri, W_tri = triangulate_all(imgLr, imgRr, P1, P2, W0)
#     if pTri.size == 0:
#         sys.exit("Triangulate no produjo puntos.")

#     T, b      = obtTyb(pTri, W_tri)
#     predTri   = pTri @ T.T + b
#     errTri    = np.linalg.norm(predTri - W_tri, axis=1)
#     print(f"[Triangulate] Afinidad base   μ={errTri.mean():.2f} mm   max={errTri.max():.2f} mm")

#     pTri_corr = systematic_correction(pTri, W_tri)
#     Tt2, bt2  = obtTyb(pTri_corr, W_tri)
#     predTri2  = pTri_corr @ Tt2.T + bt2
#     errTri2   = np.linalg.norm(predTri2 - W_tri, axis=1)
#     print(f"[Triangulate] Tras corrección μ={errTri2.mean():.2f} mm   max={errTri2.max():.2f} mm")
#     plot_3d(predTri2, W_tri, "Triangulate – GT vs Pred")

#     # ----------  MÉTODO B : MATRIZ Q ----------
#     disp = disparity_map(imgLr, imgRr)
#     Q    = build_Q_from_projections(P1_rect, P2_rect, BASELINE_MM)
#     pQ, W_Q = sample_points3D_Q(imgLr, disp, Q, W0)
#     if pQ.size == 0:
#         sys.exit("Q‑method no produjo puntos.")

#     Tq, bq   = obtTyb(pQ, W_Q)
#     predQ    = pQ @ Tq.T + bq
#     errQ     = np.linalg.norm(predQ - W_Q, axis=1)
#     print(f"[Matriz Q]   Afinidad base   μ={errQ.mean():.2f} mm   max={errQ.max():.2f} mm")

#     pQ_corr  = systematic_correction(pQ, W_Q)
#     Tq2, bq2 = obtTyb(pQ_corr, W_Q)
#     predQ2   = pQ_corr @ Tq2.T + bq2
#     errQ2    = np.linalg.norm(predQ2 - W_Q, axis=1)
#     print(f"[Matriz Q]   Tras corrección μ={errQ2.mean():.2f} mm   max={errQ2.max():.2f} mm")
#     plot_3d(predQ2, W_Q, "Matriz Q – GT vs Pred")

#     # ----------  COMPARATIVA FINAL ----------
#     print("\n=== COMPARATIVA FINAL (tras corrección sistemática) ===")
#     print(f"Triangulate  →  μ={errTri2.mean():.2f} mm   σ={errTri2.std():.2f} mm")
#     print(f"Matriz Q     →  μ={errQ2.mean():.2f} mm   σ={errQ2.std():.2f} mm")

# if __name__ == "__main__":
#     main()
import cv2, sys, os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D    # noqa: F401
from sklearn.linear_model import LinearRegression

# -----------------------------------------------------------
#  PARÁMETROS GLOBALES
# -----------------------------------------------------------
BASELINE_MM = 110.0
NUM_DISP    = 128
BLKSZ       = 5

ARUCO_DICT = {"DICT_4x4_250": cv2.aruco.DICT_4X4_250}

def detectMarkers(img, arucoDict="DICT_4x4_250"):
    dic  = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[arucoDict])
    det  = cv2.aruco.ArucoDetector(dic, cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(img)

    # Refinamiento con cornerSubPix
    if corners:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        refined_corners = []
        for corner_set in corners:
            refined = cv2.cornerSubPix(gray, corner_set, (5,5), (-1,-1), criteria)
            refined_corners.append(refined)
        corners = refined_corners

    markers = []
    if ids is not None:
        for c, i in zip(corners, ids.flatten()):
            markers.append({"id": int(i), "corners": c.reshape(4, 2)})
    return markers

def build_W0():
    A4_w, sep_x, margin, line, back_margin = 210, 200, 10, 40, 27
    W0 = np.zeros((120, 3), dtype=np.float32)
    for n in range(120):
        if n < 80:
            W0[n] = [back_margin + (n // 8) * 60,
                     2 + line + margin + (n % 8) * 50,
                     0]
        elif n < 100:
            W0[n] = [margin + 50 * ((n - 80) % 4),
                     0,
                     line + margin + 60 * ((n - 80) // 4)]
        else:
            W0[n] = [A4_w + sep_x + margin + 50 * ((n - 100) % 4),
                     0,
                     line + margin + 60 * ((n - 100) // 4)]
    return W0

OFFSETS = np.zeros((2, 4, 3), dtype=np.float32)
OFFSETS[0, 1] = [0, 40, 0];  OFFSETS[0, 2] = [40, 40, 0]; OFFSETS[0, 3] = [40, 0, 0]
OFFSETS[1, 1] = [40, 0, 0];  OFFSETS[1, 2] = [40, 0, 40]; OFFSETS[1, 3] = [0, 0, 40]

def world_corner(id_marker: int, idx_corner: int, W0):
    base = W0[id_marker]
    grp  = 0 if id_marker < 80 else 1
    return base + OFFSETS[grp, idx_corner]

def recalc_P_using_solvePnP(imgL, imgR, P1_rect, P2_rect, W0, distL, distR):
    K1, K2 = P1_rect[:, :3], P2_rect[:, :3]
    mkL, mkR = detectMarkers(imgL), detectMarkers(imgR)
    ids_common = sorted({m["id"] for m in mkL} & {m["id"] for m in mkR})
    ids_common = [i for i in ids_common if i < 120]
    if len(ids_common) == 0:
        raise RuntimeError("No hay marcadores comunes")

    objPts, imgL_pts, imgR_pts = [], [], []
    for idm in ids_common:
        mL = next(m for m in mkL if m["id"] == idm)
        mR = next(m for m in mkR if m["id"] == idm)
        for idx in range(4):
            objPts.append(world_corner(idm, idx, W0))
            imgL_pts.append(mL["corners"][idx])
            imgR_pts.append(mR["corners"][idx])

    objPts   = np.asarray(objPts,  dtype=np.float32)
    imgL_pts = np.asarray(imgL_pts, dtype=np.float32)
    imgR_pts = np.asarray(imgR_pts, dtype=np.float32)

    okL, rvecL, tvecL = cv2.solvePnP(objPts, imgL_pts, K1, distL)
    okR, rvecR, tvecR = cv2.solvePnP(objPts, imgR_pts, K2, distR)
    if not (okL and okR):
        raise RuntimeError("solvePnP falló")

    RL, _ = cv2.Rodrigues(rvecL)
    RR, _ = cv2.Rodrigues(rvecR)
    P1 = (K1 @ np.hstack((RL, tvecL))).astype(np.float32)
    P2 = (K2 @ np.hstack((RR, tvecR))).astype(np.float32)

    print(f"[solvePnP] IDs usados: {ids_common}  —  puntos: {len(objPts)}")
    return P1, P2

def triangulate_all(imgL, imgR, P1, P2, W0):
    mkL, mkR = detectMarkers(imgL), detectMarkers(imgR)
    ids_common = sorted({m["id"] for m in mkL} & {m["id"] for m in mkR})
    ids_common = [i for i in ids_common if 0 < i < 120]

    pRefs, W_pts, ids_all = [], [], []
    for idm in ids_common:
        mL = next(m for m in mkL if m["id"] == idm)
        mR = next(m for m in mkR if m["id"] == idm)
        for idx in range(4):
            cL = mL["corners"][idx].reshape(2, 1)
            cR = mR["corners"][idx].reshape(2, 1)
            X4 = cv2.triangulatePoints(P1, P2, cL, cR)
            pRefs.append((X4[:3] / X4[3]).flatten())
            W_pts.append(world_corner(idm, idx, W0))
            ids_all.append(idm)
    return np.asarray(pRefs, dtype=np.float32), np.asarray(W_pts, dtype=np.float32), np.asarray(ids_all, dtype=np.int32)

def main():
    if not os.path.exists("imgL.png") or not os.path.exists("imgR.png"):
        sys.exit("No se encuentran imgL.png / imgR.png")

    imgL = cv2.imread("imgL.png")
    imgR = cv2.imread("imgR.png")

    npz = np.load("./calibration_data960/960p/stereo_camera_calibration.npz")
    mapxL, mapyL   = npz["leftMapX"],   npz["leftMapY"]
    mapxR, mapyR   = npz["rightMapX"],  npz["rightMapY"]
    P1_rect, P2_rect = npz["leftProjection"], npz["rightProjection"]
    distL, distR = npz["leftDistortionCoeff"],npz["rightDistortionCoeff"]

    imgLr = cv2.remap(imgL, mapxL, mapyL, cv2.INTER_CUBIC)
    imgRr = cv2.remap(imgR, mapxR, mapyR, cv2.INTER_CUBIC)

    cv2.imshow("L", imgLr)
    cv2.imshow("R", imgRr)

    W0 = build_W0()

    try:
        P1, P2 = recalc_P_using_solvePnP(imgLr, imgRr, P1_rect, P2_rect, W0, distL, distR)
    except RuntimeError as e:
        print(e)
        P1, P2 = P1_rect, P2_rect

    pTri, W_tri, ids = triangulate_all(imgLr, imgRr, P1, P2, W0)

    if pTri.size == 0:
        sys.exit("Triangulate no produjo puntos.")

    mask = ids > 79
    predZ = pTri[:, 2][mask]
    gtZ   = W_tri[:, 2][mask]
    errZ  = predZ - gtZ

    print(f"\nError en Z para IDs > 79:")
    print(f"μ = {errZ.mean():.2f} mm")
    print(f"σ = {errZ.std():.2f} mm")
    print(f"max = {np.abs(errZ).max():.2f} mm")

if __name__ == "__main__":
    main()
