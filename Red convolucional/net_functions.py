import torch as t
import torch.nn as nn
import torch.optim as optim

import time
import random
import numpy as np
from tqdm import tqdm
import gc

import cnfge
from class_net import Net


# Funciones creación de instancia Net #############################################################

def rand_ADN(): # Función de creación de ADN aleatorio.

    x1, x2, x3, x4, x5 = random.sample(range(32, 64), 5) # Depth of convolutional outputs.
    y1, y2, y3, y4, y5 = random.choices(["3", "5"], k=5) # Kernel size for convolutional operation.
    z1, z2 = random.sample(range(64, 128), 2) # Size of fully connected layers.

    x1, x2, x3, x4, x5 = int(x1), int(x2), int(x3), int(x4), int(x5)
    y1, y2, y3, y4, y5 = int(y1), int(y2), int(y3), int(y4), int(y5)
    z1, z2 = int(z1), int(z2)

    batch = random.randrange(10,100)
    f = random.randint(1,4)
    lr = 1/(10)**f # 1/(10^f)

    ADN = {
        "x1":x1,"x2":x2,"x3":x3,"x4":x4,"x5":x5,
        "y1":y1,"y2":y2,"y3":y3,"y4":y4,"y5":y5,
        "z1":z1,"z2":z2,
        "BATCH_SIZE":batch,"lr":lr 
        } 
        
    print('--------------------------------------------------------------')
    print('Random ADN created.')
    print('- Conv sizes:', x1,x2,x3,x4,x5)
    print('- Kernel sizes:', y1,y2,y3,y4,y5)
    print('- Linear layer sizes:', z1,z2)
    print('- Batch size:', batch)
    print('- Learning rate:', lr)
   
    return ADN

def create_net(ADN): # Crea la red con el ADN y fija los param. de entrenamiento.

    net = Net(ADN).to(cnfge.device)
    net.optimizer = optim.Adam(net.parameters(), lr = ADN["lr"])
    net.loss_function = nn.MSELoss()

    print('--------------------------------------------------------------')
    print('Convolutional neural network created.')
    print('- Model name:', net.model_name)
    print('- ADN:', net.ADN)
    print('-', net)

    return net


# Funciones de entrenamiento y test ###############################################################

def forward_pass(net, data, label, train=False): # Pasa datos por la red y calcula acc y loss.

    if train:
        net.zero_grad()

    predicted = net(data)

    matches = [t.argmax(i) == t.argmax(j) for i, j in zip(predicted, label)]
    accuracy = matches.count(True)/len(matches)
    loss = net.loss_function(predicted, label)

    if train:
        loss.backward()
        net.optimizer.step()

    del data, label, predicted, matches
    t.cuda.empty_cache()
    
    return net, accuracy, loss


def train(net,faces): # Loop de entrenamiento.

    # Loop de entrenamiento.
    # Tiene que tener acceso a las clases net = Net() y faces = Faces().
    # De Faces() solo utiliza train_image, train_age_label y train_gender_label.

    for epoch in range(cnfge.epochs):
        
        print('--------------------------------------------------------------')
        
        running_loss = 0
        running_acc = 0
        loop_count = 0

        time.sleep(1)
        for i in tqdm (range(0, len(faces.train_image), net.batch_size)):
        
            batch_data = faces.train_image[i:i+net.batch_size]

            if True:# Data augmentation.

                for j in range(len(batch_data)): 

                    batch_data[j] = cnfge.augmentacion(batch_data[j])

            batch_data = batch_data.view(-1,1,cnfge.image_dimension,cnfge.image_dimension).to(cnfge.device)
            
            if cnfge.case == 'age':
                batch_y = faces.train_age_label[i:i+net.batch_size].to(cnfge.device)
            elif cnfge.case == 'gender':
                batch_y = faces.train_gender_label[i:i+net.batch_size].to(cnfge.device)

            net, acc, loss = forward_pass(net, batch_data, batch_y, train=True)

            del batch_data, batch_y
            t.cuda.empty_cache()

            running_loss += loss
            running_acc += acc
            loop_count += 1

        average_acc = running_acc/loop_count
        average_loss = running_loss/loop_count
        
        X, y = validate_set(faces,size=500)
        
        net, val_acc, val_loss = forward_pass(net, X, y, train=False)
        
        print(epoch,float(average_acc),float(average_loss),float(val_acc),float(val_loss))
        
                
        f = open("model.log", "a")
        f.write(f"{net.model_name},{epoch},{float(average_acc)},{float(average_loss)},{float(val_acc)},{float(val_loss)}\n")
        f.close()

        
    X, y = validate_set(faces, size=500, test = True)
    net, test_acc, test_loss = forward_pass(net, X, y, train=False)
    
    
    del X, y, faces
    t.cuda.empty_cache()

    print("PROP", t.cuda.get_device_properties(cnfge.device).total_memory/1000000)
    print("RESE", t.cuda.memory_reserved(cnfge.device)/1000000)
    print("ALLO", t.cuda.memory_allocated(cnfge.device)/1000000)

    return net, test_acc, test_loss


def validate_set(faces, size=32, test=False): # Devuelve el set de validación o test a usar.
    
    if cnfge.case == 'age':
    
        if test:
            random_start = np.random.randint(len(faces.test_image)-size)
            X, y = faces.test_image[random_start:random_start+size], faces.test_age_label[random_start:random_start+size]
        elif not test:
            random_start = np.random.randint(len(faces.validation_image)-size)
            X, y = faces.validation_image[random_start:random_start+size], faces.validation_age_label[random_start:random_start+size]
        
    elif cnfge.case == 'gender':
    
        if test:
            random_start = np.random.randint(len(faces.test_image)-size)
            X, y = faces.test_image[random_start:random_start+size], faces.test_gender_label[random_start:random_start+size]
        elif not test:
            random_start = np.random.randint(len(faces.validation_image)-size)
            X, y = faces.validation_image[random_start:random_start+size], faces.validation_gender_label[random_start:random_start+size]

    X = X.view(-1,1,cnfge.image_dimension,cnfge.image_dimension).to(cnfge.device)
    y = y.to(cnfge.device)

    del faces
    t.cuda.empty_cache()

    return X, y