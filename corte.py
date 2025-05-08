import trimesh
import numpy as np
import cv2

# Verificar que Shapely esté instalado
try:
    from shapely.geometry import Polygon
except ModuleNotFoundError:
    print("ERROR: El módulo 'shapely' no está instalado. Ejecuta 'pip install shapely' para instalarlo.")
    exit()

def draw_polygons_with_edges_opencv(polygons, img_size=512, fill_color=(0, 255, 0), edge_color=(0, 0, 0), edge_thickness=2):
    """
    Dibuja y rellena una lista de polígonos (objetos shapely.Polygon) en una imagen usando OpenCV,
    y superpone sus bordes.
    Se escala la geometría para ajustarla al tamaño de la imagen.
    """
    all_points = []
    for poly in polygons:
        pts = np.array(poly.exterior.coords)
        all_points.append(pts)
    if len(all_points) == 0:
        return 255 * np.ones((img_size, img_size, 3), dtype=np.uint8)
    
    # Combinar todos los puntos para obtener el bounding box
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
        # Rellenar el polígono
        cv2.fillPoly(img, [pts_scaled], fill_color)
        # Dibujar los bordes sobre el relleno
        cv2.polylines(img, [pts_scaled], isClosed=True, color=edge_color, thickness=edge_thickness)
    
    return img

# ----------------------------
# Configuración de la malla y puntos de corte
# ----------------------------
filename = "columna_sola.stl"  # Cambia al nombre de tu archivo STL o OBJ
mesh = trimesh.load_mesh(filename)

# Lista de puntos 3D para generar secciones
points = [
    np.array([0.0, 0.0, 0.0]),
    np.array([1.5, 2.5, 3.5]),
    np.array([2.0, 3.0, 4.0])
]

# Definir los cortes en cada plano: (vector normal, título descriptivo)
slices = [
    (np.array([1, 0, 0]), "Corte en X (plano YZ)"),
    (np.array([0, 1, 0]), "Corte en Y (plano XZ)"),
    (np.array([0, 0, 1]), "Corte en Z (plano XY)")
]

# ----------------------------
# Iterar sobre cada punto y cada plano para generar secciones y graficarlas (rellenas con bordes)
# ----------------------------
for p in points:
    for normal, title in slices:
        # Calcular la sección del mesh con el plano definido
        section = mesh.section(plane_origin=p, plane_normal=normal)
        if section is None:
            print(f"{title} - Punto {p}: Sin intersección")
            continue

        # Convertir la sección 3D a 2D utilizando to_planar()
        try:
            # to_planar() devuelve una tupla; usamos el primer elemento (Path2D)
            section_2D = section.to_planar()[0]
        except Exception as e:
            print(f"Error al convertir la sección a 2D en {title} - Punto {p}:", e)
            continue

        # Extraer la lista de polígonos completos (áreas cerradas) de la sección
        polygons = section_2D.polygons_full
        if not polygons:
            print(f"{title} - Punto {p}: No se obtuvieron polígonos")
            continue

        # Dibujar y rellenar los polígonos, superponiendo los bordes
        img = draw_polygons_with_edges_opencv(polygons, img_size=512, fill_color=(255, 255, 255), edge_color=(0, 0, 0), edge_thickness=2)
        window_name = f"{title} - Punto: {p}"
        cv2.imshow(window_name, img)
    
    # Esperar hasta que se presione una tecla para pasar al siguiente punto
    cv2.waitKey(0)
    cv2.destroyAllWindows()
