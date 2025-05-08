import trimesh
import numpy as np
import cv2

# Verificar que Shapely esté instalado
try:
    from shapely.geometry import Polygon
except ModuleNotFoundError:
    print("ERROR: El módulo 'shapely' no está instalado. Ejecuta 'pip install shapely' para instalarlo.")
    exit()

def draw_polygons_with_edges_opencv(polygons, img_size=512, fill_color=(255, 255, 255), edge_color=(0, 0, 0), edge_thickness=2):
    """
    Dibuja y rellena una lista de polígonos (objetos shapely.Polygon) en una imagen usando OpenCV,
    escalando la geometría para ajustarla al tamaño de la imagen.
    """
    all_points = []
    for poly in polygons:
        pts = np.array(poly.exterior.coords)
        all_points.append(pts)
    if len(all_points) == 0:
        return 255 * np.ones((img_size, img_size, 3), dtype=np.uint8)
    
    # Obtener el bounding box de todos los puntos
    all_points = np.vstack(all_points)
    min_val = all_points.min(axis=0)
    max_val = all_points.max(axis=0)
    diff = max_val - min_val
    diff[diff == 0] = 1  # Evitar división por cero

    margin = 20
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    
    for poly in polygons:
        pts = np.array(poly.exterior.coords)
        # Escalar y trasladar los puntos a coordenadas de píxeles
        pts_scaled = ((pts - min_val) / diff) * (img_size - 2 * margin) + margin
        pts_scaled = pts_scaled.astype(np.int32)
        pts_scaled = pts_scaled.reshape((-1, 1, 2))
        # Rellenar el polígono y dibujar el borde
        cv2.fillPoly(img, [pts_scaled], fill_color)
        cv2.polylines(img, [pts_scaled], isClosed=True, color=edge_color, thickness=edge_thickness)
    
    return img

def graficar_cortes_stl(posicion, ruta_stl, img_size=512):
    """
    Grafica los cortes (secciones) de una malla STL a partir de una posición 3D.
    
    Parámetros:
      - posicion: numpy array con la coordenada 3D [x, y, z] para generar los cortes.
      - ruta_stl: cadena de texto con la ruta al archivo STL.
      - img_size: tamaño de la imagen para la visualización (por defecto 512).
      
    Se generan cortes en tres planos:
      - Corte en X (plano YZ)
      - Corte en Y (plano XZ)
      - Corte en Z (plano XY)
      
    Cada corte se muestra en una ventana en una posición fija y distinta,
    y se actualiza inmediatamente cada vez que se invoque la función.
    """
    # Cargar la malla STL
    mesh = trimesh.load_mesh(ruta_stl)
    
    # Definir los cortes: (vector normal, título descriptivo)
    cortes = [
        (np.array([1, 0, 0]), "Corte en X (plano YZ)"),
        (np.array([0, 1, 0]), "Corte en Y (plano XZ)"),
        (np.array([0, 0, 1]), "Corte en Z (plano XY)")
    ]
    
    # Definir posiciones fijas para cada ventana
    window_positions = {
        "Corte en X (plano YZ)": (100, 100),
        "Corte en Y (plano XZ)": (600, 100),
        "Corte en Z (plano XY)": (1100, 100)
    }
    
    # Para cada corte se calcula la sección y se actualiza la ventana correspondiente
    for normal, titulo in cortes:
        section = mesh.section(plane_origin=posicion, plane_normal=normal)
        if section is None:
            print(f"{titulo} - Sin intersección en la posición {posicion}")
            continue
        try:
            section_2D = section.to_planar()[0]
        except Exception as e:
            print(f"Error al convertir la sección a 2D en {titulo}: {e}")
            continue
        polygons = section_2D.polygons_full
        if not polygons:
            print(f"{titulo} - No se obtuvieron polígonos en la posición {posicion}")
            continue
        
        img = draw_polygons_with_edges_opencv(polygons, img_size=img_size)
        cv2.namedWindow(titulo, cv2.WINDOW_NORMAL)
        pos = window_positions[titulo]
        cv2.moveWindow(titulo, pos[0], pos[1])
        cv2.imshow(titulo, img)
    
    # Espera corta para actualizar la GUI sin detener la ejecución (1 ms)
    cv2.waitKey(1)

if __name__ == '__main__':
    # Ejemplo de uso de la función graficar_cortes_stl:
    # Se define una posición de ejemplo; reemplázala por la posición real obtenida.
    posicion_ejemplo = np.array([1.0, 2.0, 3.0])
    ruta_stl = "columna_sola.stl"
    
    # Ejecución en un bucle para simular actualizaciones de posición
    import time
    for i in range(10):
        # Se puede modificar la posición en cada iteración
        nueva_posicion = posicion_ejemplo + np.array([0.1 * i, 0.1 * i, 0.1 * i])
        graficar_cortes_stl(nueva_posicion, ruta_stl)
        # Se añade un pequeño retardo para observar la actualización (por ejemplo, 100 ms)
        time.sleep(0.1)
    
    # Al final se espera hasta que el usuario presione alguna tecla para cerrar las ventanas.
    cv2.waitKey(0)
    cv2.destroyAllWindows()
