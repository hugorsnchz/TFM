import torch as t
import torch.nn as nn
import torch.nn.functional as F
import time
import cnfge

# Clase de Pytorch para formar la arquitectura. ###################################################

class Net(nn.Module):
    def __init__(self, ADN):
        super().__init__()

        image_dimension = cnfge.image_dimension 
        image_layers = cnfge.image_layers
        output_size = cnfge.output_size
        
        self.model_name = f"model-{int(time.time())}"
        self.ADN = ADN
        self.loss_function = []
        self.optimizer = []
        self.batch_size = ADN["BATCH_SIZE"]

        # Asignación de variables ADN a variables locales.

        x1, x2, x3, x4, x5 = ADN["x1"], ADN["x2"], ADN["x3"], ADN["x4"], ADN["x5"]    # Tamaño de conv
        y1, y2, y3, y4, y5 = ADN["y1"], ADN["y2"], ADN["y3"], ADN["y4"], ADN["y5"]    # Tamaño del kérnel
        z1, z2 = ADN["z1"], ADN["z2"]                                                 # Tamaño de lineales

        # Construcción de las capas convolucionales.
        # padding_mode = 'zeros' por defecto.

        self.convolutional_1 = nn.Conv2d(image_layers, x1, y1, padding=round((y1-1)/2)) 
        self.convolutional_1_id = nn.Conv2d(image_layers, x1, 1)

        self.convolutional_2 = nn.Conv2d(x1, x1+x2, y2, padding=round((y2-1)/2))
        self.convolutional_2_id = nn.Conv2d(x1, x1+x2, 1)

        self.convolutional_3 = nn.Conv2d(x1+x2, x1+x2+x3, y3, padding=round((y3-1)/2))
        self.convolutional_3_id = nn.Conv2d(x1+x2, x1+x2+x3, 1)

        self.convolutional_4 = nn.Conv2d(x1+x2+x3, x1+x2+x3+x4, y4, padding=round((y4-1)/2))
        self.convolutional_4_id = nn.Conv2d(x1+x2+x3, x1+x2+x3+x4, 1)

        self.convolutional_5 = nn.Conv2d(x1+x2+x3+x4, x1+x2+x3+x4+x5, y5, padding=round((y5-1)/2))
        self.convolutional_5_id = nn.Conv2d(x1+x2+x3+x4, x1+x2+x3+x4+x5, 1)

        # Cálculo de la salida convolucional.

        example = t.randn(image_layers, image_dimension, image_dimension).view(
            -1, image_layers, image_dimension, image_dimension)

        self.linear = None
        self.convolucionales(example)

        # Construcción de las capas neuronales.

        self.full_connect_1 = nn.Linear(self.linear, z1)
        self.full_connect_2 = nn.Linear(z1, z2)
        self.full_connect_3 = nn.Linear(z2, output_size)

    def convolucionales(self, x_0): # 'Forward' de las capas convolucionales.

        # 1. init -> inicial, primera convolución.
        # 2. res -> residual, convolución identidad (kernel 1x1).
        # 3. Suma y pool.
        # 4. red -> reducido. Usado para la concatenación posterior.

        x_1_init= F.relu(self.convolutional_1(x_0))
        x_1_res = self.convolutional_1_id(x_0)
        x_1 = F.max_pool2d(x_1_init + x_1_res, (2, 2))
        x_1_red = F.max_pool2d(x_1_init + x_1_res, (3, 3))

        x_2_init = F.relu(self.convolutional_2(x_1))
        x_2_res = self.convolutional_2_id(x_1)
        x_2 = F.max_pool2d(x_2_init + x_2_res, (2, 2))
        x_2_red = F.max_pool2d(x_2_init + x_2_res, (3, 3))
        
        x_3_init = F.relu(self.convolutional_3(x_2))
        x_3_res = self.convolutional_3_id(x_2)
        x_3 = F.max_pool2d(x_3_init + x_3_res, (2, 2))
        x_3_red = F.max_pool2d(x_3_init + x_3_res, (3, 3))

        x_4_init = F.relu(self.convolutional_4(x_3))
        x_4_res = self.convolutional_4_id(x_3)
        x_4 = F.max_pool2d(x_4_init + x_4_res, (2, 2))
        x_4_red = F.max_pool2d(x_4_init + x_4_res, (3, 3))

        x_5_init = F.relu(self.convolutional_5(x_4))
        x_5_res = self.convolutional_5_id(x_4)
        # x_5 = F.max_pool2d(x_5_init + x_5_res, (2, 2))
        x_5_red = F.max_pool2d(x_5_init + x_5_res, (3, 3))

        # Transformación a vectores y concatenación.

        x_1_flat = t.flatten(x_1_red, start_dim=1)
        x_2_flat = t.flatten(x_2_red, start_dim=1)
        x_3_flat = t.flatten(x_3_red, start_dim=1)
        x_4_flat = t.flatten(x_4_red, start_dim=1)
        x_5_flat = t.flatten(x_5_red, start_dim=1)

        x_6 = t.cat([x_1_flat, x_2_flat, x_3_flat, x_4_flat, x_5_flat], 1)

        if self.linear is None:
            self.linear = (x_6.shape[1])

        return x_6

    def full_connected(self, x_6): # 'Forward' de las capas neuronales.

        x_7 = F.relu(self.full_connect_1(x_6))
        x_8 = F.relu(self.full_connect_2(x_7))
        x_9 = F.softmax(self.full_connect_3(x_8), dim=1)
        
        return x_9

    def forward(self, x_0): # Función 'forward'.
        
        x_6 = self.convolucionales(x_0)
        x_9 = self.full_connected(x_6)

        return x_9