import numpy as np
import matplotlib.pyplot as plt

def update(coords, axes):
    """
    Dada una lista de 4 puntos 3D (coords), calcula:
      - El punto con menor suma de distancias (punto "cercano")
      - El punto con mayor suma (punto "lejano")
      - El promedio de los dos puntos intermedios
    Luego:
      1. Calcula la recta primaria (entre cercano y lejano).
      2. Calcula el vector normal al plano formado por esa recta y la dirección hacia el punto intermedio.
      3. Desplaza la recta 2 cm en esa dirección.
      4. Extiende simétricamente la recta desplazada 10 cm (5 cm extra en cada extremo).
      5. Grafica ambas rectas (original desplazada y extendida) en los planos XY, XZ y YZ.
    """
    # Limpiar subplots y fijar límites
    for ax in axes:
        ax.clear()
        ax.set_xlim(-1, 25)
        ax.set_ylim(-1, 25)
        ax.grid(True)
    
    if len(coords) >= 4 and all(c is not None for c in coords):
        points = np.array(coords)  # Espera una matriz de 4x3
        
        # Calcular suma de distancias de cada punto a los demás
        dists = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        sum_dists = dists.sum(axis=1)
        
        # Índices: punto cercano (mínima suma) y punto lejano (máxima suma)
        i_min = np.argmin(sum_dists)  # Punto "cercano"
        i_max = np.argmax(sum_dists)  # Punto "lejano"
        
        # Los dos puntos restantes (intermedios)
        all_indices = set(range(len(points)))
        remaining = list(all_indices - {i_min, i_max})
        if len(remaining) != 2:
            print("Error: Se requieren exactamente 4 puntos.")
            return
        
        # Promedio de los dos puntos intermedios
        p_inter = (points[remaining[0]] + points[remaining[1]]) / 2.0
        
        # Vector de la recta primaria (de cercano a lejano)
        v = points[i_max] - points[i_min]
        # Vector desde el punto cercano al intermedio
        v2 = p_inter - points[i_min]
        
        # Calcular el vector normal al plano definido por v y v2
        n = np.cross(v, v2)
        norm_n = np.linalg.norm(n)
        if norm_n == 0:
            print("No se pudo calcular un vector normal válido.")
            return
        n_normalized = n / norm_n
        
        # Desplazar la recta 2 cm en la dirección normal
        displacement = 2  # 2 cm
        p1_disp = points[i_min] + displacement * n_normalized
        p2_disp = points[i_max] + displacement * n_normalized
        
        # --- Extensión del segmento ---
        # Calcular la dirección de la recta desplazada
        seg_vector = p2_disp - p1_disp
        seg_length = np.linalg.norm(seg_vector)
        if seg_length == 0:
            print("El segmento desplazado tiene longitud 0.")
            return
        u = seg_vector / seg_length
        
        extension = 20  # Alargar 10 cm en total
        # Extensión simétrica: 5 cm en cada extremo
        p1_ext = p2_disp 
        p2_ext = p2_disp +10.5* u
        
        # --- Graficado en cada plano ---
        # Plano XY
        axes[0].plot([p1_disp[0], p2_disp[0]], [p1_disp[1], p2_disp[1]], 
                     'm*-', linewidth=2, label='Original Desplazada')
        axes[0].plot([p1_ext[0], p2_ext[0]], [p1_ext[1], p2_ext[1]], 
                     'c.-', linewidth=2, label='Segmento Extendido')
        axes[0].set_title("Plano XY")
        axes[0].set_xlabel("X")
        axes[0].set_ylabel("Y")
        axes[0].legend()
        
        # Plano XZ
        axes[1].plot([p1_disp[0], p2_disp[0]], [p1_disp[2], p2_disp[2]], 
                     'm*-', linewidth=2, label='Original Desplazada')
        axes[1].plot([p1_ext[0], p2_ext[0]], [p1_ext[2], p2_ext[2]], 
                     'c.-', linewidth=2, label='Segmento Extendido')
        axes[1].set_title("Plano XZ")
        axes[1].set_xlabel("X")
        axes[1].set_ylabel("Z")
        axes[1].legend()
        
        # Plano YZ
        axes[2].plot([p1_disp[1], p2_disp[1]], [p1_disp[2], p2_disp[2]], 
                     'm*-', linewidth=2, label='Original Desplazada')
        axes[2].plot([p1_ext[1], p2_ext[1]], [p1_ext[2], p2_ext[2]], 
                     'c.-', linewidth=2, label='Segmento Extendido')
        axes[2].set_title("Plano YZ")
        axes[2].set_xlabel("Y")
        axes[2].set_ylabel("Z")
        axes[2].legend()
    
    plt.tight_layout()
    plt.pause(0.001)
