#!/usr/bin/env python3
# ------------------------------------------------------------
# Cálculo de error vs. circunferencia ideal  (todo en mm)
# ------------------------------------------------------------
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

# ---------- (1) Trazo ideal ----------
e,h = 0.5,50        # cm
l   = 50 - e
alt = np.sqrt(h**2 - l**2)
theta = (np.pi/2-np.arccos(l/h))
d   = 20
c_cm = np.array([32,
                 (50-d)*np.cos(theta),
                 l - np.sqrt((50-d)**2 - alt**2)])
r_cm = 11

Rx = np.array([[1,0,0],
               [0, np.cos(-theta), -np.sin(-theta)],
               [0, np.sin(-theta),  np.cos(-theta)]])
normal = Rx@np.array([0,0,1])
v1 = np.cross(normal,[1,0,0])
if np.linalg.norm(v1)==0:
    v1 = np.cross(normal,[0,1,0])
v1/=np.linalg.norm(v1); v2=np.cross(normal,v1)

t = np.linspace(0,2*np.pi,400)
circle_mm = (c_cm + r_cm*(np.cos(t)[:,None]*v1 + np.sin(t)[:,None]*v2) + 0.5*normal)*10  # → mm
tree      = cKDTree(circle_mm)


# 1) Cargo los errores
ext2_mm = np.load('trayectoria_ext2_mm.npy')   # Nx3
dists,_ = tree.query(ext2_mm)

# 2) Métricas robustas ya vistas
median_err = np.median(dists)
mad        = np.median(np.abs(dists - median_err))  # Median Absolute Deviation

# 3) Defino umbral basado en MAD (p. ej. k=3)
k          = 3
threshold  = median_err + k*mad
mask_good  = dists <= threshold
mask_bad   = ~mask_good

print(f"Umbral MAD (k={k}): {threshold:.2f} mm")
print(f"Outliers detectados: {mask_bad.sum()} de {len(dists)} puntos")

# 4) Métricas antes y después de filtrar
rmse_all    = np.sqrt((dists**2).mean())
rmse_filtered = np.sqrt((dists[mask_good]**2).mean())

print(f"RMSE total          : {rmse_all:.2f} mm")
print(f"RMSE sin outliers   : {rmse_filtered:.2f} mm")
print(f"Mediana error       : {median_err:.2f} mm")
print(f"MAD                 : {mad:.2f} mm")

# 5) Visualizar histogramas
plt.figure(figsize=(8,4))
plt.hist(dists, bins=50, alpha=0.6, label='Todos')
plt.hist(dists[mask_good], bins=50, alpha=0.6, label='Filtrados')
plt.axvline(threshold, color='k', linestyle='--', label=f'Umbal {threshold:.1f}mm')
plt.xlabel('Error (mm)')
plt.ylabel('Frecuencia')
plt.legend()
plt.title('Errores: antes y después de filtrar outliers')
plt.show()

# 6) Opcional: ver índices de los peores outliers
outlier_indices = np.where(mask_bad)[0]
print("Primeros 10 índices con mayor error:", outlier_indices[:10])

# 7) Visualización 3D de los puntos y la circunferencia
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

# Trazo medido
ax.scatter(ext2_mm[:,0], ext2_mm[:,1], ext2_mm[:,2], c='tab:blue', s=2, label='Trayectoria medida')

# Circunferencia ideal desplazada
ax.plot(circle_mm[:,0], circle_mm[:,1], circle_mm[:,2], c='tab:red', lw=2, label='Circunferencia ideal')

ax.set_xlabel('X [mm]')
ax.set_ylabel('Y [mm]')
ax.set_zlabel('Z [mm]')
ax.set_title('Comparación 3D: trayectoria vs. circunferencia ideal')
ax.legend()
ax.view_init(elev=30, azim=45)  # Ajusta vista 3D si lo necesitás
plt.tight_layout()
plt.show()
