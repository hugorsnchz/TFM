import torch

# Solución de errores #############################################################################

# "RuntimeError: error in LoadLibraryA" t.cat
import ctypes
ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll') 

import matplotlib
matplotlib.use('TkAgg', force=True)

# Aceleración GPU #################################################################################

global device

# [Falta imprimir características de la GPU usada.]
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU.")
else:
    device = torch.device("cpu")
    print("Running on the CPU.")

# Parámetros ######################################################################################

global image_dimension, image_layers, output_size, age_span

image_dimension = 32    # Dimensión cuadrada plana de la imagen inicial.
image_layers = 3        # Número de capas (1 = B&W, 3 = RGB).
output_size = 10        # Dimensión  del vector de salida de la red.
epochs = 10
