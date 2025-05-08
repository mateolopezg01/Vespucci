import trimesh
import cv2
import numpy as np

def compute_mapping_info(mesh, image_size):
    """
    Calcula la transformación (escala y offset) entre el sistema de
    coordenadas del modelo (proyección en el plano XZ) y las coordenadas
    de la imagen, basándose en el bounding box de la malla.
    
    Parámetros:
      - mesh: objeto trimesh de la malla STL.
      - image_size: (ancho, alto) de la imagen (en píxeles).
    
    Retorna:
      - mapping_info: diccionario con 'scale', 'offset', 'bounding_box' y 'image_size'.
    """
    # Extraer las coordenadas (x,z) de cada vértice
    vertices_xz = mesh.vertices[:, [0, 2]]
    min_xz = vertices_xz.min(axis=0)
    max_xz = vertices_xz.max(axis=0)
    
    # Calcular la escala en X y Z para ajustar el bounding box a la imagen
    scale_x = image_size[0] / (max_xz[0] - min_xz[0])
    scale_z = image_size[1] / (max_xz[1] - min_xz[1])
    scale = min(scale_x, scale_z)
    
    # El offset traslada el mínimo del bounding box a (0,0)
    offset = -min_xz  
    mapping_info = {
        "scale": scale,
        "offset": offset,
        "bounding_box": (min_xz, max_xz),
        "image_size": image_size
    }
    return mapping_info

def world_to_image(pt, mapping_info):
    """
    Convierte un punto (x,z) del mundo a coordenadas de imagen para una vista frontal.
    Se invierte la coordenada X para obtener la vista opuesta.
    
    Parámetros:
      - pt: tupla o array (x, z) del mundo.
      - mapping_info: diccionario con la información de mapeo.
    
    Retorna:
      - [x_img, y_img]: coordenadas del píxel en la imagen.
    """
    scale = mapping_info["scale"]
    offset = mapping_info["offset"]
    image_width, image_height = mapping_info["image_size"]
    
    # Invertir la X para vista frontal
    x_img = image_width - int((pt[0] + offset[0]) * scale)
    # Mapeo directo de Z, invirtiendo la vertical para que el origen esté en la esquina superior izquierda
    y_img = image_height - int((pt[1] + offset[1]) * scale)
    return [x_img, y_img]

def image_to_world(pixel, mapping_info):
    """
    Función inversa: dada una posición de píxel (x,y) en la imagen, 
    calcula la coordenada (x,z) correspondiente en el mundo.
    
    Parámetros:
      - pixel: tupla (x, y) en la imagen.
      - mapping_info: diccionario con la información de mapeo.
    
    Retorna:
      - (x_world, z_world): coordenadas en el mundo.
    """
    scale = mapping_info["scale"]
    offset = mapping_info["offset"]
    image_width, image_height = mapping_info["image_size"]
    
    # Invertir la transformación en X (por la vista frontal)
    x_world = (image_width - pixel[0]) / scale - offset[0]
    z_world = (image_height - pixel[1]) / scale - offset[1]
    return (x_world, z_world)

def mouse_callback(event, x, y, flags, param):
    """
    Callback de OpenCV que, al hacer clic en la imagen, utiliza la información
    de mapeo para convertir la posición del píxel a la coordenada (x,z) del modelo.
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        mapping_info = param["mapping_info"]
        world_coord = image_to_world((x, y), mapping_info)
        print(f"Pixel ({x}, {y}) -> World (x,z): {world_coord}")
        # Marcar el punto en la imagen
        cv2.circle(param["image"], (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Registered Image", param["image"])

if __name__ == '__main__':
    # Ruta a la imagen PNG ya guardada
    png_path = "screen.png"  # Reemplaza con la ruta de tu imagen
    image = cv2.imread(png_path)
    if image is None:
        print("No se pudo cargar la imagen:", png_path)
        exit(1)
    
    # Obtener el tamaño de la imagen (ancho, alto)
    height, width, channels = image.shape
    image_size = (width, height)
    
    # Cargar la malla STL usando trimesh (para calcular el mapping)
    stl_path = "columna_sola.stl"  # Reemplaza con la ruta a tu archivo STL
    mesh = trimesh.load_mesh(stl_path)
    
    # Calcular la información de mapeo basada en la proyección (x,z) y el bounding box
    mapping_info = compute_mapping_info(mesh, image_size)
    
    # Mostrar la imagen y configurar el callback para registrar clics
    cv2.namedWindow("Registered Image")
    cv2.imshow("Registered Image", image)
    callback_params = {"mapping_info": mapping_info, "image": image.copy()}
    cv2.setMouseCallback("Registered Image", mouse_callback, callback_params)
    
    print("Haz clic en la imagen para obtener la coordenada (x,z) correspondiente.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
