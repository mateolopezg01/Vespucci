import numpy as np
import matplotlib.pyplot as plt


data = np.load('videos/prefs.npz')

# Si solo hay un array, podrías hacer:
array_unico = data[data.files[0]]

print(array_unico)
# Cerrar el archivo .npz
data.close()