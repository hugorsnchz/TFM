import torch as t
import cv2
import time
import pickle
import numpy as np
import scipy.io
from tqdm import tqdm
import cnfge
import random

###################################################################################################
# Clase que carga los datos del orgien, los transforma y ordena en sus respectivos sets de 'train',
# 'validation' y 'test' para el posterior entrenamiento.

class Faces():

    ###############################################################################################
    # Inicializa la clase y establece sus variables (self).

    def __init__(self):     
        
        # Pretratamiento.
        self.training_data = []
        
        # Post tratamiento.
        self.treated_training_data = []
        self.images_tensor = []
        self.label_age_onehot = []
        self.label_gender_onehot = []

        # Sets de entrenamiento.
        self.train_image = []
        self.train_age_label = []
        self.train_gender_label  =[]
        
        # Sets de prueba.
        self.test_image = []
        self.test_age_label = []
        self.test_gender_label = []
        
        # Sets de validación.
        self.validation_image = []
        self.validation_age_label = []
        self.validation_gender_label = []
        
        print('--------------------------------------------------------------')
        print('Data class Faces() created.')


    ###############################################################################################
    # Cuenta el número de datos en función de la edad y el género, identificando aquellos erróneos.
    # Trabaja con los datos de la variable 'self.training_data'.

    def create_report(self, pretratamiento):
        
        print('--------------------------------------------------------------')
        print('Creating report...')

        # Datos para el recuento.
        if pretratamiento == True:
            gender = self.training_data[:,3]
            age = self.training_data[:,4]
            training_data = self.training_data

        elif pretratamiento == False:
            gender = self.treated_training_data[:,3]
            age = self.treated_training_data[:,4]
            training_data = self.treated_training_data
        
        # Definición de los contadores.
        gender_count = np.zeros((2,), dtype=int)
        age_count = np.zeros((int(100/cnfge.age_span),), dtype=int)
        age_error = np.zeros((2,), dtype=int)
        gender_error = 0
        
        # Condiciones de conteo.
        time.sleep(1)
        for i in range(gender.size):
            
            if age[i] > 100:
                age_error[0] += 1
                #print('age:', self.age[self.count])
                
            elif age[i] <= 0:
                age_error[1] += 1
                #print('age:', self.age[self.count])
                
            else:
                age_count[int((age[i]-1)/cnfge.age_span)] += 1
    
            if (gender[i] != 1) and (gender[i] != 0):
                gender_error += 1
                #print('gender:', self.gender[self.count])
    
            else:
                gender_count[int(gender[i])] += 1
        
        print("- Size of data set:", len(training_data))
        print("- Men:", gender_count[0])
        print("- Women:", gender_count[1])
        print("- Gender errors:", gender_error)
        print("- Age distribution 0-100:", age_count)
        print("- Age over 100:", age_error[0])
        print("- Negative age:", age_error[1])
        print('Creating finished.')
    

    ###############################################################################################
    # Carga los datos de edad, género y nombre del .mat original y carga las imágenes en blanco y negro
    # desde el directorio correspondiente a la variable 'training_data'.

    def make_training_data(self):
        
        print('--------------------------------------------------------------')
        print('Loading metadata...')
        
        # Establece dimensión de la imagen y directorio de localización.
        image_dimension = cnfge.image_dimension
        directorio = "wiki_crop/"
               
        # Extrae la información del fichero .mat.
        mat = scipy.io.loadmat("wiki_crop_with_age.mat")
        mat = mat["wiki"]
        mat = np.asarray(mat[0])
        
        # Definición de los contadores.
        count = 0
        error = []
        
        # Extracción de los datos relevantes a variables de la clase.
        full_path = mat[0][2][0]
        gender = mat[0][3][0]
        name = mat[0][4][0]
        age = mat[0][8][0]

        c = list(zip(full_path, gender, name, age))
        random.shuffle(c)
        full_path, gender, name, age = zip(*c)
        
        # Extracción del resto de variables no relevntes.
        # dob = mat[0][0][0]
        # photo_taken = mat[0][1][0]
        # face_location = mat[0][5][0]
        # face_score = mat[0][6][0]
        # second_face_score = mat[0][7][0]
        
        print('Metadata loaded.')
        
        print('--------------------------------------------------------------')
        print("Fetching image data from directory...")
             
        # Bucle sobre todas las rutas de directorio.
        time.sleep(1)
        for f in tqdm(full_path):
            
            # Formateo de la ruta de las imágenes.
            f = str(f)
            f = f[2:len(f)-2]
            f = directorio + f

            try:
                # Lectura, redimensionamiento y paso a escala de grises.
                image = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (image_dimension, image_dimension))
                self.training_data.append([count, np.array(image), name[count], gender[count], age[count]])

            except Exception as e:
                error.append([f,e])
                pass

            count = count + 1
        
        print('Image data fetched with', len(error), 'exceptions.')
        
        print('--------------------------------------------------------------')
        print('Shuffling and saving data...')
        
        # Mezclado de los datos y paso a array.
        self.training_data = np.array(self.training_data)
        np.random.shuffle(self.training_data)

        # Guardado de la variable al directorio.
        with open("training_data.pickle", 'wb') as f:
            pickle.dump(self.training_data, f)
            
        print('Data saved to "training_data.pickle".')
        
        self.training_data = []
        self.treated_training_data = []
        self.images_tensor = []
        self.label_age_onehot = []
        self.label_gender_onehot = []
        self.train_image = []
        self.train_age_label = []
        self.train_gender_label  =[]
        self.test_image = []
        self.test_age_label = []
        self.test_gender_label = []
        self.validation_image = []
        self.validation_age_label = []
        self.validation_gender_label = []


    ###############################################################################################
    # Carga desde el archivo training_data.pickle la variable 'self.training_data'.

    def load_training_data(self, pretratamiento):
        
        if pretratamiento == True:
            
            print('--------------------------------------------------------------')
            print ("Loading data from training_data.pickle...")
            with open("training_data.pickle", 'rb') as f:
                self.training_data = pickle.load(f)
            print ('Data loaded. Size:', len(self.training_data))
    
        elif pretratamiento == False:
    
            print('--------------------------------------------------------------')
            print ("Loading data from treated_training_data.pickle...")
            with open("treated_training_data.pickle", 'rb') as f:
                self.treated_training_data = pickle.load(f)
            print ('Data loaded. Size:', len(self.treated_training_data))
          

    ###############################################################################################
    # Trata y transforma la variable 'self.training_data' en los tensores y etiquetas 'one-hot' 
    # correspondientes. Luego forma los sets de 'train', 'test' y 'validation'.

    def treat_training_data(self):

        training_data = self.training_data
        image_dimension = cnfge.image_dimension
        age_span = cnfge.age_span
        indices = []
        
        print('--------------------------------------------------------------')
        print("Treating data...")
        
        # Ajuste de labels +100 o negativos
        older, negative = 0, 0
        
        time.sleep(2)
        
        for i in tqdm(range(len(training_data))):
            
            if training_data[i,4] > 100:
                training_data[i,4] = 100
                older += 1
                
            elif training_data[i,4] <= 0:
                training_data[i,4] = 1
                negative += 1
                
        time.sleep(2)      
        
        print("- Negative ages as 1:", negative)
        print("- Ages decreased to 100:", older)
        
        # Balance de datos. 
        age_count = np.zeros((int(100/cnfge.age_span),), dtype=int)

        time.sleep(2)
        
        for i in tqdm(range(len(training_data))):
            age_count[int((training_data[i,4]-1)/cnfge.age_span)] += 1
        
        time.sleep(2)
        
        median = np.median(age_count)
        median = 2000
        print('- Labels median:',median)

        age_count = np.zeros((int(100/cnfge.age_span),), dtype=int)
        balance_count = 0
        
        time.sleep(2)
        
        for i in tqdm(range(len(training_data))):
            age_count[int((training_data[i,4]-1)/cnfge.age_span)] += 1
            if age_count[int((training_data[i,4]-1)/cnfge.age_span)] >= median:
             indices.append(i)
             balance_count += 1
             
        time.sleep(2)
        
        print('- Discarded labels for balance:',balance_count)

        # Detección de géneros erróneos
        wrong_gender = 0
        
        time.sleep(2)
        
        for i in tqdm(range(len(training_data))):
            
            if np.isnan(training_data[i,3]):
                wrong_gender += 1
                indices.append(i)
                
        time.sleep(2)
        
        print("- Discarded gender labels:", wrong_gender)
        
        # Extracción de labels redondeados.
        label_gender = np.array([i[3] for i in training_data]).astype(int)
        label_age = np.array([i[4] for i in training_data])
        label_age = np.floor((label_age-1)/age_span).astype(int)
       
        # Normalización de las imagenes (0,1).
        images_tensor = t.Tensor([i[1] for i in training_data]).view(-1,image_dimension,image_dimension)
        images_tensor = images_tensor/255.0
        
        # Detección de imágenes negras y blancas.
        dark, bright = 0, 0 
        
        time.sleep(2)
        
        for i in tqdm(range(len(images_tensor))):
            
            if np.average(images_tensor[i]) > 0.95:
                #print("Imagen clara, índice:", i)
                indices.append(i)
                bright +=  1
                
            elif np.average(images_tensor[i]) < 0.05:
                #print("Imagen oscura, índice:", i)
                indices.append(i)  
                dark += 1
        
        time.sleep(2)
        
        print("- Discarded dark images:", dark)
        print("- Discarded bright images:", bright)

        # Eliminación de los índices detectados erroneos.
        label_gender = np.delete(label_gender,indices)
        label_age = np.delete(label_age,indices)
        images_np = images_tensor.numpy()
        images_np = np.delete(images_np, indices, axis=0)
        training_data = np.delete(training_data, indices, axis=0)

        # Creación de labels one-hot.
        label_age_onehot = np.zeros((label_age.size, label_age.max()+1))
        label_age_onehot[np.arange(label_age.size),label_age] = 1
        label_gender_onehot = np.zeros((label_gender.size, label_gender.max()+1))
        label_gender_onehot[np.arange(label_gender.size),label_gender] = 1
        
        # Transformación a tensores.
        label_gender_onehot = t.Tensor(label_gender_onehot)
        label_age_onehot = t.Tensor(label_age_onehot)
        images_tensor = t.Tensor(images_np)
        
        # Actualización de variables self
        self.treated_training_data = training_data
        self.label_age_onehot = label_age_onehot
        self.label_gender_onehot = label_gender_onehot
        self.images_tensor = images_tensor

        print("Data ready. One-hot enconded labels made.")
        
        print('--------------------------------------------------------------')
        print("Saving treated data to treated_training_data.pickle...")
        
        with open("treated_training_data.pickle", 'wb') as f:
            pickle.dump(self.treated_training_data, f)
            
        print('treated_training_data saved.')
        
        print('--------------------------------------------------------------')
        print('Saving labels to "labels.pickle" and image tensors to "image_tensors.pickle".')
        
        with open("labels.pickle", 'wb') as f:
            pickle.dump([self.label_age_onehot, self.label_gender_onehot], f)

        with open("image_tensors.pickle", 'wb') as f:
            pickle.dump(self.images_tensor, f)
            
        print('Labels and image tensors saved.')

        self.training_data = []
        self.treated_training_data = []
        self.images_tensor = []
        self.label_age_onehot = []
        self.label_gender_onehot = []
        self.train_image = []
        self.train_age_label = []
        self.train_gender_label  =[]
        self.test_image = []
        self.test_age_label = []
        self.test_gender_label = []
        self.validation_image = []
        self.validation_age_label = []
        self.validation_gender_label = []

    def make_sets(self):
        
        print('--------------------------------------------------------------')
        print("Loading labels and image tensors...")

        with open("labels.pickle", 'rb') as f:
            [self.label_age_onehot, self.label_gender_onehot] = pickle.load(f)

        with open("image_tensors.pickle", 'rb') as f:
            self.images_tensor = pickle.load(f)

        print("Labels and image tensors loaded.")

        print('--------------------------------------------------------------')
        print("Creating sets...")

        c = list(zip(self.images_tensor,self.label_age_onehot,self.label_gender_onehot))
        random.shuffle(c)
        self.images_tensor,self.label_age_onehot,self.label_gender_onehot = zip(*c)

        self.images_tensor = t.stack(list(self.images_tensor), dim=0)
        self.label_age_onehot = t.stack(list(self.label_age_onehot), dim=0)
        self.label_gender_onehot = t.stack(list(self.label_gender_onehot), dim=0)

        validation_percentage = 0.15
        test_percentage = 0.15
        
        val_size = int(len(self.images_tensor)*validation_percentage)
        test_size = int(len(self.images_tensor)*test_percentage)
        train_size = len(self.images_tensor)-(val_size + test_size)

        train_image = self.images_tensor[:train_size]
        train_age_label = self.label_age_onehot[:train_size]
        train_gender_label = self.label_gender_onehot[:train_size]
        print("- Size of training sets", len(train_image),len(train_age_label),len(train_gender_label))

        test_image = self.images_tensor[train_size:-val_size]
        test_age_label = self.label_age_onehot[train_size:-val_size]
        test_gender_label = self.label_gender_onehot[train_size:-val_size]
        print("- Size of test ses:", len(test_image), len(test_age_label), len(test_gender_label))

        validation_image = self.images_tensor[-val_size:]
        validation_age_label = self.label_age_onehot[-val_size:]
        validation_gender_label = self.label_gender_onehot[-val_size:]
        print("- Size of validation sets:", len(validation_image), len(validation_age_label), len(validation_gender_label))
       
        print("Sets created. Total size:", len(train_image)+len(test_image)+len(validation_image))

        print('--------------------------------------------------------------')
        print('Saving sets of images...')

        with open("image_sets.pickle", 'wb') as f:
            pickle.dump([train_image, test_image, validation_image], f)
            
        print('Saving sets of age labels...')

        with open("age_sets.pickle", 'wb') as f:
            pickle.dump([train_age_label, test_age_label, validation_age_label], f)
            
        print('Saving sets of gender labels...')

        with open("gender_sets.pickle", 'wb') as f:
            pickle.dump([train_gender_label, test_gender_label, validation_gender_label], f)        
            
        print("Sets saved to .pickle archives.")
        
        self.training_data = []
        self.treated_training_data = []
        self.images_tensor = []
        self.label_age_onehot = []
        self.label_gender_onehot = []
        self.train_image = []
        self.train_age_label = []
        self.train_gender_label  =[]
        self.test_image = []
        self.test_age_label = []
        self.test_gender_label = []
        self.validation_image = []
        self.validation_age_label = []
        self.validation_gender_label = []
        
    ###############################################################################################
    # Carga los sets seleccionados desde los archivos .pickle correspondientes.
    # Carga la variable self.training_data desde el archivo .pickle para poder rehacer el report.
    
    def load_sets(self, images=False, age=False, gender=False): 
        
        print('--------------------------------------------------------------')
        print("Loading sets from .pickle archives...")
        
        if images == True:
            with open("image_sets.pickle", 'rb') as f:
                [train_image, test_image, validation_image] = pickle.load(f)
                
            self.train_image = train_image
            self.test_image = test_image
            self.validation_image = validation_image
            print("Image sets loaded.")
            
        if age == True:
            with open("age_sets.pickle", 'rb') as f:
               [train_age_label, test_age_label, validation_age_label] = pickle.load(f)
            
            self.train_age_label = train_age_label
            self.test_age_label = test_age_label
            self.validation_age_label = validation_age_label
            print("Age label sets loaded.")
            
        if gender == True:
            with open("gender_sets.pickle", 'rb') as f:
                [train_gender_label, test_gender_label, validation_gender_label] = pickle.load(f)

            self.train_gender_label  =train_gender_label
            self.test_gender_label = test_gender_label
            self.validation_gender_label = validation_gender_label
            print("Gender label sets loaded.")
        
        print("Finished loading sets.")