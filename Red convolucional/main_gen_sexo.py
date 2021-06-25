from class_faces import Faces
import net_gen_functions as ngf
import cnfge

cnfge.case = 'gender'
cnfge.output_size = 2

# Creación de la clase de datos y tratamiento.

if not True:
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
    
# Loop de entrenamiento generacional.

ngf.generacion(faces, generaciones = 20, individuos = 10)