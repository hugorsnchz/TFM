import torch.nn as nn
import torch.optim as optim

import cnfge
from class_net import Net


# Funciones creación de instancia Net #############################################################

def create_net(ADN): # Crea la red con el ADN y fija los param. de entrenamiento.

    net = Net(ADN).to(cnfge.device)
    net.optimizer = optim.Adam(net.parameters(), lr = ADN["lr"])
    net.loss_function = nn.CrossEntropyLoss()

    print('--------------------------------------------------------------')
    print('Convolutional neural network created.')
    print('- Model name:', net.model_name)
    print('- Procesor:', cnfge.device)
    print('- ADN:', net.ADN)
    print('-', net)

    return net
