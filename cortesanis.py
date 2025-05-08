import os
import json
import trimesh
import numpy as np
import cv2
from shapely.affinity import translate
def align_section(section, plane_normal, up_vector=(0, 1, 0)):
    """
    Aplica una transformación a 'section' para alinear el plano de corte de modo que:
      - La normal del plano se convierta en el eje Z.
      - El vector 'up_vector' se convierta en el eje Y.
    Esto garantiza que la orientación 2D resultante sea consistente.
    """
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    up_vector = np.array(up_vector) / np.linalg.norm(up_vector)
    
    # Eje X local: producto cruzado entre up_vector y plane_normal
    new_x = np.cross(up_vector, plane_normal)
    new_x /= np.linalg.norm(new_x)
    
    # Eje Y local: producto cruzado entre plane_normal y new_x
    new_y = np.cross(plane_normal, new_x)
    new_y /= np.linalg.norm(new_y)
    
    new_z = plane_normal

    # Matriz de transformación (4x4)
    transform = np.array([
        [new_x[0], new_y[0], new_z[0], 0],
        [new_x[1], new_y[1], new_z[1], 0],
        [new_x[2], new_y[2], new_z[2], 0],
        [0,        0,        0,        1]
    ])

    section.apply_transform(transform)
    return section

i=0
def draw_polygons_with_edges_opencv(polygons, img_size=512, 
                                    fill_color=(255, 255, 255), 
                                    edge_color=(0, 0, 0), 
                                    edge_thickness=2):
    """
    Dibuja y rellena una lista de polígonos en una imagen usando OpenCV, 
    escalando la geometría para ajustarla al tamaño de la imagen.
    Retorna la imagen y un diccionario con los parámetros de transformación.
    """
    all_points = []
    for poly in polygons:
        pts = np.array(poly.exterior.coords)
        all_points.append(pts)
    if len(all_points) == 0:
        return 255 * np.ones((img_size, img_size, 3), dtype=np.uint8), None
    
    all_points = np.vstack(all_points)
    min_val = all_points.min(axis=0)
    max_val = all_points.max(axis=0)
    diff = max_val - min_val
    diff[diff == 0] = 1  # Evitar división por cero

    margin = 20
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    for poly in polygons:
        pts = np.array(poly.exterior.coords)
        pts_scaled = ((pts - min_val) / diff) * (img_size - 2 * margin) + margin
        pts_scaled=np.array(pts_scaled)
        pts_scaled = pts_scaled.astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(img, [pts_scaled], fill_color)
        cv2.polylines(img, [pts_scaled], isClosed=True, color=edge_color, thickness=edge_thickness)
    
    mapping = {
        "margin": margin,
        "min_val": min_val.tolist(),
        "diff": diff.tolist(),
        "img_size": img_size
    }
    return img, mapping

def save_slices_along_z(mesh, x_fixed, y_fixed, z_start, z_end, step, img_size, output_dir):
    """
    Genera cortes de la malla en el plano XY (normal [0,0,1]) variando la coordenada z.
    Se alinea cada sección para obtener una orientación consistente y se desplaza el origen
    del sistema de coordenadas 2D de los polígonos, de forma que la proyección del punto [0,0,z]
    quede en (0,0).
    
    Retorna un diccionario con los mappings de cada corte.
    
    Se asume que existen las funciones 'align_section' y 
    'draw_polygons_with_edges_opencv' para alinear la sección y dibujar los polígonos.
    """
    mappings = {}
    normal = np.array([0, 0, 1])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    z_values = np.arange(z_start, z_end + step/2, step)
    for z in z_values:
        # Definimos el origen para el corte en [x_fixed, y_fixed, z]
        origin = np.array([0, 0, z])
        section = mesh.section(plane_origin=origin, plane_normal=normal)
        if section is None:
            print(f"No hay intersección para z = {z:.2f}")
            continue
        
        try:
            # Alineamos la sección para que tenga una orientación consistente.
            # Se asume que 'align_section' está definida, por ejemplo, para que 'up' sea (0, -1, 0)
            section = align_section(section, normal, up_vector=(0, -1, 0))
            
            # Proyectamos la sección a 2D y obtenemos la matriz de transformación T_planar.
            section_2D, T_planar = section.to_planar()
            
        except Exception as e:
            print(f"Error al convertir sección para z = {z:.2f}: {e}")
            continue

                   # Proyectar el punto [0, 0, z] usando la transformación T_planar.
            # Es importante pasar el punto como array de forma (1, 3)
        projected_origin = trimesh.transform_points(np.array([[0, 0, z]]), T_planar)[0][:2]
            
            # Desplazar cada polígono para que la proyección del punto [0, 0, z] se sitúe en (0,0)
        shifted_polygons = [translate(poly, xoff=projected_origin[0], yoff=projected_origin[1])
                                for poly in section_2D.polygons_full]
 
        
        if not shifted_polygons:
            print(f"No se obtuvieron polígonos para z = {z:.2f}")
            continue
        
        # Dibujar la imagen de la sección usando la función 'draw_polygons_with_edges_opencv'
        img, mapping = draw_polygons_with_edges_opencv(shifted_polygons, img_size=img_size)
        filename = os.path.join(output_dir, f"slice_z_{z:06.2f}.png")
        cv2.imwrite(filename, img)
        print(f"Guardado corte en z = {z:.2f} en {filename}")
        
        if mapping is not None:
            mappings[f"slice_z_{z:06.2f}"] = mapping

    return mappings


def save_slices_along_x(mesh, y_fixed, z_fixed, x_start, x_end, step, img_size, output_dir):
    """
    Genera cortes de la malla en el plano YZ (normal [1,0,0]) variando la coordenada x.
    Se alinea cada sección para mantener una orientación consistente y se desplaza el origen
    del sistema de coordenadas 2D de los polígonos, de forma que la proyección del punto [x,0,0]
    quede en (0,0).
    
    Retorna un diccionario con los mappings de cada corte.
    
    NOTA: Se asume que existen las funciones 'align_section' y 
    'draw_polygons_with_edges_opencv' para alinear la sección y dibujar los polígonos.
    """
    mappings = {}
    normal = np.array([1, 0, 0])
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Genera los valores de x para los cortes
    x_values = np.arange(x_start, x_end + step/2, step)
    
    for x in x_values:
        # Definir el origen del plano: en este ejemplo se usa [x, 0, 0]
        origin = np.array([x, 0, 0])
        
        # Se obtiene la sección 3D de la malla
        section = mesh.section(plane_origin=origin, plane_normal=normal)
        if section is None:
            print(f"No hay intersección para x = {x:.2f}")
            continue
        
        try:
            # Alineamos la sección para que tenga una orientación consistente.
            # Se asume que 'align_section' está definida, por ejemplo, para que 'up' sea (0, -1, 0)
            section = align_section(section, normal, up_vector=(0, -1, 0))
            
            # Se proyecta la sección a 2D. 'section.to_planar()' retorna un tuple:
            # (section_2D, T_planar), donde T_planar es la matriz de transformación usada.
            section_2D, T_planar = section.to_planar()
        except Exception as e:
            print(f"Error al convertir sección para x = {x:.2f}: {e}")
            continue
        
        # Proyectar el punto [x, 0, 0] al sistema 2D usando la transformación T_planar.
        # Es importante pasar el punto como un array de forma (1,3).
        projected_origin = trimesh.transform_points(np.array([[x, 0, 0]]), T_planar)[0][:2]
        
        # Desplazar cada polígono para que la proyección del punto [x, 0, 0] se mueva a (0,0)
        shifted_polygons = [translate(poly, xoff=projected_origin[0], yoff=projected_origin[1])
                            for poly in section_2D.polygons_full]
        
        if not shifted_polygons:
            print(f"No se obtuvieron polígonos para x = {x:.2f}")
            continue
        
        # Dibujar la imagen de la sección usando la función 'draw_polygons_with_edges_opencv'
        # que se asume que está definida. Esta función debe retornar la imagen y un mapping.
        img, mapping = draw_polygons_with_edges_opencv(shifted_polygons, img_size=img_size)
        filename = os.path.join(output_dir, f"slice_x_{x:06.2f}.png")
        cv2.imwrite(filename, img)
        print(f"Guardado corte en x = {x:.2f} en {filename}")
        
        if mapping is not None:
            mappings[f"slice_x_{x:06.2f}"] = mapping

    return mappings
if __name__ == '__main__':
    # Ruta al archivo STL
    ruta_stl = "columna_sola.stl"  # Reemplaza con tu archivo STL
    # Cargar la malla
    mesh = trimesh.load_mesh(ruta_stl)
    
    # Definir resolución (tamaño de imagen) para los cortes
    img_size = 512  # Imagen de 512x512 píxeles
    
    # Usar el centro de la malla para definir las posiciones fijas
    bounds = mesh.bounds  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
    center = bounds.mean(axis=0)
    
    # Cortes en el eje Z (plano XY): x fijo = center_x, y fijo = center_y, z varía
    # x_fixed = center[0]
    # y_fixed = center[1]
    z_start = 0.0
    z_end = 110.0
    step = 0.5
    output_dir_z = "slices_z"
    print("Generando cortes a lo largo del eje Z...")
    mappings_z = save_slices_along_z(mesh, 0, 0, z_start, z_end, step, img_size, output_dir_z)
    
    # Cortes en el eje X (plano YZ): y fijo = center_y, z fijo = center_z, x varía
    y_fixed_x = center[1]
    z_fixed_x = center[2]
    print("y fijo:", y_fixed_x,"z fijo:", z_fixed_x)
    x_start = -42.0
    x_end = 46.0
    output_dir_x = "slices_x"
    print("Generando cortes a lo largo del eje X...")
    mappings_x = save_slices_along_x(mesh, 0, 0, x_start, x_end, step, img_size, output_dir_x)
    
    # Combinar los mappings de ambos ejes en un solo diccionario
    all_mappings = {
        "slices_z": mappings_z,
        "slices_x": mappings_x
    }
    
    # Guardar todos los mappings en un único archivo JSON
    with open("all_mappings.json", "w") as f:
        json.dump(all_mappings, f, indent=4)
    print("Todos los mappings han sido guardados en 'all_mappings.json'.")
    
    print("Generación de cortes finalizada.")
