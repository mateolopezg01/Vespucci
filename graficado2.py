import cv2
import json
import numpy as np
from segmentardo import extremos_segmento
# Cargar el archivo único de mappings solo una vez
with open("all_mappings.json", "r") as f:
    all_mappings = json.load(f)

# Crear las tres ventanas una sola vez
cv2.namedWindow("Corte en X (YZ)", cv2.WINDOW_NORMAL)
cv2.namedWindow("Corte en Y (XZ)", cv2.WINDOW_NORMAL)
cv2.namedWindow("Corte en Z (XY)", cv2.WINDOW_NORMAL)
cv2.moveWindow("Corte en X (YZ)", 0, 100)
cv2.moveWindow("Corte en Y (XZ)", 500, 100)
cv2.moveWindow("Corte en Z (XY)", 1000, 100)

def coord_to_img(y, z, mapping):
    """
    Transforma coordenadas (y, z) usando el mapping proporcionado para el corte en X (vista YZ).
    """
    img_size = mapping["img_size"]
    diff = np.array(mapping["diff"])
    margin = mapping["margin"]
    min_val = np.array(mapping["min_val"])
    pt = ((np.array([y, z]) - min_val) / diff) * (img_size - 2 * margin) + margin
    return pt


def world_to_image_y(pt, image_size_y=[512, 512], scale=4.100035471925065, offset=[52.82499695, 1.97956085]):
    """
    Transforma un punto del mundo (x, z) a coordenadas de imagen.
    Se invierte la coordenada x para obtener la vista frontal.
    """
    x_img = image_size_y[0] - int((pt[0] + offset[0]) * scale)
    y_img = image_size_y[1] - int((pt[1] + offset[1]) * scale)
    return [x_img, y_img]

def graficado2(coordenadas):
    ext1, ext2=extremos_segmento(coordenadas)
    # print("Ext1:", ext1)
    # print("Ext2",ext2)
    (x, y, z) = ext2
    
    x_rounded = round(x * 2) / 2
    z_rounded = round(z * 2) / 2

    # Determinar la clave para el corte en X (vista YZ)
    if x_rounded < -42:
        key_x = "slice_x_-42.00"
    elif x_rounded > 46:
        key_x = "slice_x_46.00"
    else:
        key_x = f"slice_x_{x_rounded:06.2f}"

    # Determinar la clave para el corte en Z (vista XY)
    if z_rounded < 0:
        key_z = "slice_z_000.00"
    elif z_rounded > 110:
        key_z = "slice_z_110.00"
    else:
        key_z = f"slice_z_{z_rounded:06.2f}"

    # Rutas de imagen basadas en las claves
    path_imx = f"slices_x/{key_x}.png"
    path_imz = f"slices_z/{key_z}.png"

    # Obtener los mappings correspondientes desde all_mappings
    mapping_x = all_mappings["slices_x"].get(key_x)
    mapping_z = all_mappings["slices_z"].get(key_z)

    if mapping_x is None or mapping_z is None:
        # print("Mapping no encontrado para la clave correspondiente.")
        return

    # Cargar las imágenes de cada corte
    corte_x = cv2.imread(path_imx)
    corte_z = cv2.imread(path_imz)
    vista_y = cv2.imread("superior.png")

    if corte_x is None or corte_z is None or vista_y is None:
        print("Error al cargar alguna imagen.")
        return

    # Dibujar línea en el corte en X (vista YZ) utilizando mapping_x
    pt_x = coord_to_img(ext1[2], -ext1[1], mapping_x)
    pt_x2 = coord_to_img(ext2[2], -ext2[1], mapping_x)
    cv2.line(corte_x, (round(pt_x[0]), round(pt_x[1])), (round(pt_x2[0]), round(pt_x2[1])), (0, 0, 255), 3)
    # cv2.circle(corte_x,(round(pt_x[0]), round(pt_x[1])), 5, (0, 0, 255), -1)
    # Dibujar marcador en la imagen superior (vista frontal)
    pt_world = [x, z]
    x_y, z_y = world_to_image_y(pt_world)
    cv2.circle(vista_y, (x_y, z_y), 5, (0, 0, 255), -1)

    # Dibujar línea en el corte en Z (vista XY) utilizando mapping_z
    # pt_z = coord_to_img(x, y, mapping_z)
    pt_z = coord_to_img(ext1[0], -ext1[1], mapping_z)
    pt_z2 = coord_to_img(ext2[0], -ext2[1], mapping_z)
    cv2.line(corte_z, (round(pt_z[0]), round(pt_z[1])), (round(pt_z2[0]), round(pt_z2[1])), (0, 0, 255), 3)
    # cv2.circle(corte_z, (round(pt_z[0]), round(pt_z[1])), 5, (0, 0, 255), -1)
    # Actualizar el contenido de cada ventana sin destruirlas
    cv2.imshow("Corte en X (YZ)", corte_x)
    cv2.imshow("Corte en Y (XZ)", vista_y)
    cv2.imshow("Corte en Z (XY)", corte_z)

    # Espera corta para permitir que la GUI se actualice
    cv2.waitKey(1)
