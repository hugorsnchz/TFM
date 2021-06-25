import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision as tv
from torchvision import transforms, datasets

import os
import cv2
import time
import math
import random
import pickle
import numpy as np
import scipy.io
from tqdm import tqdm
import gc

import net_functions as nf


# Funciones genéticas #############################################################################

def generacion(faces, generaciones = 2, individuos = 8):
    
    tiempo = round(time.time(),4)

    f = open("log.log", "a")
    f.write(f"\n\n")
    f.write(f" ############################################### {tiempo}\n\n")
    f.close()

    f = open("best.log", "a")
    f.write(f"\n\n")
    f.write(f" ############################################### {tiempo}\n\n")
    f.close()

    dtype = [('name', np.unicode_, 16), ('ADN', list), ('acc', float), ('loss', float)]
    resultado = np.array([], dtype=dtype)
    best = np.array([], dtype=dtype)

    
    poblacion = []
    
    for individuo in range(individuos):
        aleatorio = nf.rand_ADN()
        poblacion = np.append(poblacion, aleatorio)


    for generacion in range(generaciones):
        
        resultado = np.array([], dtype=dtype)
    
        for individuo in range(len(poblacion)):
            
            print('--------------------------------------------------------------')
            print("- Generacion", generacion+1)
            print("- Individuo", individuo+1)
            
            ADN = poblacion[individuo]
            net = nf.create_net(ADN)
            net, test_acc, test_loss = nf.train(net, faces)
            
            f = open("log.log", "a")
            f.write(f"{generacion+1},{individuo+1},{net.model_name},{test_acc},{test_loss} \n")
            f.close()

            resultado = np.append(resultado, np.array([(net.model_name, ADN, float(test_acc), float(test_loss))], dtype=resultado.dtype))

            s = r'C:\Users\hugor\Google Drive\Universidad\Máster en Automática y Robótica\TFM\Red neuronal\new try\modelos' + '\\'+ net.model_name + ".pickle"
            s = r"/home/hrajado/Documents/TFM HRS/Code/v4/modelos"+ '/'+ net.model_name + ".pickle"
            
            with open(s, 'wb') as f:
                pickle.dump(net, f)
                    
            del net, ADN
            t.cuda.empty_cache()

        resultado = np.sort(resultado, order=['acc'])
        best = resultado[-1]
        padres = resultado[-4:]
        ADN_padre1, ADN_madre1 = padres[0][1], padres[1][1]
        ADN_padre2, ADN_madre2 = padres[2][1], padres[3][1] # mejores en acc
        ADN_hijo1, ADN_hija1 = rep_ADN(ADN_padre1, ADN_madre1)
        ADN_hijo2, ADN_hija2 = rep_ADN(ADN_padre2, ADN_madre2)

        f = open("best.log", "a")
        f.write(f"{generacion+1},{best} \n")
        f.close()
        
        poblacion = [ADN_hijo1, ADN_hija1, ADN_padre2, ADN_madre2, ADN_hijo2, ADN_hija2]

        for i in range(individuos-len(poblacion)):
            aleatorio = nf.rand_ADN()
            poblacion = np.append(poblacion, aleatorio)

        resultado = np.array([], dtype=dtype)

    print("Fin del algoritmo evolutivo.")
    return best

def rep_ADN(ADN_padre, ADN_madre): # Función que combina dos ADN padres y obtiene dos ADN hijos.

    ADN_hijo = {}
    ADN_hija = {}

    for x in ADN_padre: # Cruce
        
        eleccion = round(random.random()) 

        if eleccion == 0:
            ADN_hijo[x] = ADN_padre[x]
            ADN_hija[x] = ADN_madre[x]
        elif eleccion == 1:
            ADN_hija[x] = ADN_padre[x]
            ADN_hijo[x] = ADN_madre[x]
        else:
            print("Error de combinación.")

    for y in ADN_hija, ADN_hijo: # Mutación

        for x in y: 
            if isinstance(y[x], int) and x not in ["y1", "y2", "y3", "y4", "y5", "lr", "BATCH_SIZE"]:
                y[x] = round(y[x]+y[x]*random.uniform(-0.2, 0.2))

    return ADN_hijo, ADN_hija