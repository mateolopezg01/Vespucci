import trimesh
import cv2
import numpy as np

def look_at_matrix(eye, target, up):
    """
    Calcula una matriz de vista (world-to-camera) usando el método lookAt.
    Luego, para asignarla a la cámara de trimesh (que espera camera-to-world),
    se deberá invertir la matriz.
    """
    eye = np.array(eye, dtype=np.float64)
    target = np.array(target, dtype=np.float64)
    up = np.array(up, dtype=np.float64)
    
    # Vector hacia adelante (f)
    f = target - eye
    f /= np.linalg.norm(f)
    
    # Lado (s): producto cruz de f y up
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    
    # Nuevo up (u)
    u = np.cross(s, f)
    
    M = np.eye(4, dtype=np.float64)
    M[0, :3] = s
    M[1, :3] = u
    M[2, :3] = -f
    M[0, 3] = -np.dot(s, eye)
    M[1, 3] = -np.dot(u, eye)
    M[2, 3] = np.dot(f, eye)
    return M

def compute_mapping_info(mesh, image_size):
    """
    Calcula la transformación (escala y offset) entre el sistema de 
    coordenadas del modelo (proyección en el plano XZ) y las coordenadas
    de imagen, basándose en el bounding box de la malla.
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
    Se invierte la coordenada X para obtener la vista frontal.
    """
    scale = mapping_info["scale"]
    offset = mapping_info["offset"]
    image_width, image_height = mapping_info["image_size"]
    
    # pt[0] es x, pt[1] es z
    x_img = image_width - int((pt[0] + offset[0]) * scale)
    y_img = image_height - int((pt[1] + offset[1]) * scale)
    return [x_img, y_img]

def image_to_world(pixel, mapping_info):
    """
    Función inversa: dada una posición de píxel (x,y) en la imagen, 
    calcula la coordenada (x,z) correspondiente en el mundo.
    """
    scale = mapping_info["scale"]
    offset = mapping_info["offset"]
    image_width, image_height = mapping_info["image_size"]
    
    x_world = (image_width - pixel[0]) / scale - offset[0]
    z_world = (image_height - pixel[1]) / scale - offset[1]
    return (x_world, z_world)

def get_original_image(mesh, image_size):
    """
    Genera una imagen usando el renderizado básico de trimesh, configurando
    la cámara para obtener una vista frontal en el plano (x,z). Se usa fondo negro.
    """
    scene = trimesh.Scene(mesh)
    # Calcular la matriz de vista (world-to-camera) y luego invertirla
    M = look_at_matrix(eye=[0, 200, 0], target=[0, 0, 60], up=[0, 0, 1])
    transform = np.linalg.inv(M)
    scene.camera_transform = transform

    # Renderizar la imagen con fondo negro (RGBA: [0,0,0,255])
    png_bytes = scene.save_image(resolution=image_size, background=[0, 0, 0, 255])
    img_array = np.asarray(bytearray(png_bytes), dtype=np.uint8)
    original_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return original_img

def mouse_callback(event, x, y, flags, param):
    """
    Callback de OpenCV: al hacer clic se convierte la posición del píxel a la
    coordenada (x,z) usando el mapping y se marca el punto en la imagen.
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        mapping_info = param["mapping_info"]
        world_coord = image_to_world((x, y), mapping_info)
        print(f"Pixel ({x}, {y}) -> World (x,z): {world_coord}")
        cv2.circle(param["image"], (x, y), 5, (0, 0, 255), -1)
        cv2.imshow(param["window_name"], param["image"])

if __name__ == '__main__':
    # Definir la resolución a usar (debe coincidir en ambas imágenes)
    image_size = (720, 720)
    
    # Rutas: STL y PNG registrado
    stl_path = "columna_sola.stl"            # Reemplaza con la ruta a tu STL
    registered_png = "screen.png"   # Reemplaza con la ruta a tu imagen PNG
    
    # Cargar la malla usando trimesh
    mesh = trimesh.load_mesh(stl_path)
    
    # Calcular el mapping usando las coordenadas (x,z)
    mapping_info = compute_mapping_info(mesh, image_size)
    
    # Obtener la imagen original (renderizada con trimesh en vista frontal)
    original_img = get_original_image(mesh, image_size)
    
    # Cargar la imagen registrada
    reg_img = cv2.imread(registered_png)
    if reg_img is None:
        print("No se pudo cargar la imagen registrada:", registered_png)
        exit(1)
    
    # Mostrar ambas imágenes en ventanas separadas
    cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Registered Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Original Image", original_img)
    cv2.imshow("Registered Image", reg_img)
    
    # Configurar callbacks para ambas ventanas
    orig_params = {"mapping_info": mapping_info, "image": original_img.copy(), "window_name": "Original Image"}
    reg_params  = {"mapping_info": mapping_info, "image": reg_img.copy(), "window_name": "Registered Image"}
    cv2.setMouseCallback("Original Image", mouse_callback, orig_params)
    cv2.setMouseCallback("Registered Image", mouse_callback, reg_params)
    
    print("Haz clic en cualquiera de las imágenes para obtener la coordenada (x,z) correspondiente.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
