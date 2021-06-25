import pickle
from class_faces import Faces
import net_functions as nf
import cnfge

cnfge.case = 'gender'
cnfge.output_size = 2

# Definición de los parámetros de la red.

ADN = {
        "x1":50,"x2":50,"x3":50,"x4":50,"x5":50,
        "y1":3,"y2":3,"y3":3,"y4":3,"y5":3,
        "z1":128,"z2":256,
        "BATCH_SIZE":50,"lr":0.000001 
        } 


# Creación de la red con los parámetros especificados.

net = nf.create_net(ADN)

# Creación de la clase de datos y su tratamiento.

if  True:
    faces = Faces()
    faces.make_training_data()
    faces.load_training_data(pretratamiento=True)
    faces.create_report(pretratamiento=True)
    faces.treat_training_data()
    faces.load_training_data(pretratamiento=False)
    faces.create_report(pretratamiento=False)
    faces.make_sets()
    faces.load_sets(images=True, age=False, gender=True)
    
else:
    faces = Faces()
    faces.load_sets(images=True, age=False, gender=True)
    

# Loop de entrenamiento.

net, final_acc, final_loss = nf.train(net, faces)

print('--------------------------------------------------------------')
print('Final accuracy: ', float(final_acc)) 
print('Final loss: ', float(final_loss))

# Guardado de la red entrenada.

s = r'C:\Users\hugor\Google Drive\Universidad\Máster en Automática y Robótica\TFM\Red neuronal\new try\modelos' + '\\'+ net.model_name + "_s.pickle"           
with open(s, 'wb') as f:
    pickle.dump(net, f)

# with open("net1.pickle", 'rb') as f:
#         net = pickle.load(f)
