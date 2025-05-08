import trimesh
import numpy as np
import cv2
import json

def stl_to_frontal_image(stl_path, image_size=(512, 512)):
    """
    Genera una vista frontal (vista opuesta a la posterior) del modelo STL en
    modo ortográfico (proyección sobre el plano XZ) y calcula la transformación
    entre coordenadas del mundo (x,z) e imagen.

    En esta vista se invierte la coordenada X para mostrar la vista opuesta.

    Parámetros:
      - stl_path: Ruta al archivo STL.
      - image_size: (ancho, alto) de la imagen generada.

    Retorna:
      - img: Imagen generada (numpy array).
      - mapping_info: Diccionario con los parámetros para transformar entre
                      coordenadas del mundo (x,z) e imagen.
    """
    # Cargar la malla STL
    mesh = trimesh.load_mesh(stl_path)
    
    # Extraer las coordenadas (x,z) de los vértices
    vertices_xz = mesh.vertices[:, [0, 2]]
    
    # Calcular el bounding box en el plano XZ
    min_xz = vertices_xz.min(axis=0)
    max_xz = vertices_xz.max(axis=0)
    
    print("Min ",min_xz,"max ",max_xz)
    # Crear una imagen en blanco (fondo blanco)
    img = np.ones((image_size[1], image_size[0], 3), dtype=np.uint8) * 255
    
    # Calcular el factor de escala en cada dirección y usar el menor para conservar la relación de aspecto
    scale_x = image_size[0] / (max_xz[0] - min_xz[0])
    scale_z = image_size[1] / (max_xz[1] - min_xz[1])
    scale = min(scale_x, scale_z)
    print("Scale: ", scale)
    # Definir el offset para trasladar el mínimo a (0,0)
    offset = -min_xz  # [offset_x, offset_z]
    print("Offset: ", offset)
    def world_to_image(pt):
        """
        Transforma un punto del mundo (x,z) a coordenadas de imagen.
        Se invierte la coordenada x para obtener la vista frontal.
        """
        # En la vista frontal, se invierte el eje x:
        x_img = image_size[0] - int((pt[0] + offset[0]) * scale)
        # Para z, se invierte la coordenada vertical (ya que el origen de imagen es la esquina superior izquierda)
        y_img = image_size[1] - int((pt[1] + offset[1]) * scale)
        return [x_img, y_img]
    
    # Dibujar cada cara (triángulo) de la malla en la imagen
    for face in mesh.faces:
        pts = mesh.vertices[face][:, [0, 2]]  # Extraer (x,z)
        pts_img = np.array([world_to_image(pt) for pt in pts], dtype=np.int32)
        pts_img = pts_img.reshape((-1, 1, 2))
        cv2.fillPoly(img, [pts_img], color=(200, 200, 200))
    
    # Dibujar los bordes de los triángulos
    for face in mesh.faces:
        pts = mesh.vertices[face][:, [0, 2]]
        pts_img = np.array([world_to_image(pt) for pt in pts], dtype=np.int32)
        pts_img = pts_img.reshape((-1, 1, 2))
        cv2.polylines(img, [pts_img], isClosed=True, color=(0, 0, 0), thickness=1)
    
    mapping_info = {
        "scale": scale,
        "offset": offset,
        "bounding_box": (min_xz, max_xz),
        "image_size": image_size,
        "world_to_image": world_to_image,
        # Para la transformación inversa, se debe tener en cuenta la inversión en x.
        "inversion": True
    }
    
    return img, mapping_info

def image_to_world(pixel, mapping_info):
    """
    Función inversa: dada la posición de un píxel en la imagen, calcula la
    coordenada (x,z) correspondiente en el sistema de coordenadas del modelo.
    
    En la vista frontal se invierte la transformación en X.
    
    Parámetros:
      - pixel: Tupla (x, y) en la imagen.
      - mapping_info: Diccionario obtenido de stl_to_frontal_image().
    
    Retorna:
      - (x_world, z_world): Coordenadas en el plano XZ del modelo.
    """
    scale = mapping_info["scale"]
    offset = mapping_info["offset"]
    image_width, image_height = mapping_info["image_size"]
    
    # Para x se debe invertir la transformación:
    x_world = (image_width - pixel[0]) / scale - offset[0]
    z_world = (image_height - pixel[1]) / scale - offset[1]
    return (x_world, z_world)

def click_event(event, x, y, flags, param):
    """
    Callback que se activa al hacer clic en la imagen.
    Convierte la posición del píxel en coordenadas (x,z) y las muestra por consola.
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        mapping_info = param["mapping_info"]
        world_coord = image_to_world((x, y), mapping_info)
        print(f"Pixel {(x, y)} corresponde a la coordenada (x,z): {world_coord}")
        # Marcar el punto en la imagen
        cv2.circle(param["image"], (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Vista Frontal", param["image"])

if __name__ == '__main__':
    stl_path = "columna_sola.stl"  # Reemplaza con la ruta a tu archivo STL
    image_size = (512, 512)
    
    # Generar la vista frontal y obtener la información de mapeo
    img, mapping_info = stl_to_frontal_image(stl_path, image_size=image_size)
    # print(mapping_info)
    # with open("superior.json", "w") as archivo:
    #     json.dump(mapping_info, archivo, indent=4)
    # Crear una copia para mostrar y marcar los puntos
    img_display = img.copy()
    
    cv2.namedWindow("Vista Frontal")
    cv2.imshow("Vista Frontal", img_display)
    params = {"mapping_info": mapping_info, "image": img_display}
    cv2.setMouseCallback("Vista Frontal", click_event, params)
    cv2.imwrite("superior.png", img_display)

    print("Haga clic en la imagen para obtener la coordenada (x,z) correspondiente (vista frontal).")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
