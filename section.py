import trimesh
import numpy as np
from shapely.affinity import translate
import matplotlib.pyplot as plt

# Crear una malla de prueba, por ejemplo, una esfera
mesh = trimesh.creation.icosphere(subdivisions=3)

# Definir un plano de corte (usamos la normal Z para un corte horizontal)
plane_origin = mesh.centroid
plane_normal = [0, 0, 1]

# Obtener la sección 3D del mesh
section = mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)

if section is None:
    print("No se encontró intersección con el plano.")
else:
    # Proyectar la sección a 2D
    section_2D, T = section.to_planar()

    # Obtener los polígonos completos (cada uno es un objeto shapely.geometry.Polygon)
    polygons = section_2D.polygons_full

    # Definir la traslación deseada en 2D (por ejemplo, 1 unidad en X y 2 unidades en Y)
    dx, dy = 1.0, 2.0

    # Aplicar la traslación a cada polígono usando Shapely
    moved_polygons = [translate(poly, xoff=dx, yoff=dy) for poly in polygons]

    # Mostrar por consola las coordenadas de los polígonos trasladados
    for i, poly in enumerate(moved_polygons):
        print(f"Polígono {i}: {list(poly.exterior.coords)}")

    # Para visualizar la diferencia, graficamos los polígonos originales y trasladados
    fig, ax = plt.subplots()
    
    # Graficar los polígonos originales en azul
    for poly in polygons:
        x, y = poly.exterior.xy
        ax.plot(x, y, 'b-', label='Original' if 'Original' not in ax.get_legend_handles_labels()[1] else "")
    
    # Graficar los polígonos trasladados en rojo
    for poly in moved_polygons:
        x, y = poly.exterior.xy
        ax.plot(x, y, 'r--', label='Trasladado' if 'Trasladado' not in ax.get_legend_handles_labels()[1] else "")
    
    ax.set_aspect('equal')
    ax.legend()
    plt.title("Comparación de polígonos originales y trasladados")
    plt.show()
