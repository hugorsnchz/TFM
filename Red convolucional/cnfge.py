import torch as t
import torchvision as tv
from torchvision import transforms

# Aceleración GPU #################################################################################

global device

print('--------------------------------------------------------------')

if t.cuda.is_available():
    device = t.device("cuda:0")
    print("Running on the GPU.")
else:
    device = t.device("cpu")
    print("Running on the CPU.")

prop = t.cuda.get_device_properties(device)
print(device, prop.name, prop.total_memory)
    
# Parámetros ######################################################################################

global image_dimension, image_layers, output_size, age_span, epochs, case

case = 'none'
image_dimension = 64                # Dimensión cuadrada plana de la imagen inicial.
image_layers = 1                    # Número de capas (1 = B&W, 3 = RGB).
age_span = 10                       # En años. Rango de edad de los grupos. Factor de 100 (1,2,4,5,10,20,25,50).
output_size = int(100/age_span)     # Dimensión  del vector de salida de la red.
epochs = 150

# Composición de operaciones para el aumento de variedad de los datos.

augmentacion = tv.transforms.Compose([transforms.ToPILImage(), 
                                      transforms.RandomCrop(round(image_dimension*0.9)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.Resize(image_dimension),
                                      transforms.ToTensor()])
