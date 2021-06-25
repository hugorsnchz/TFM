import face_recognition
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import pickle
import os
import time

os.system("cls") # Limpia command line.

# Localiza las carpetas de fotos.
init_folder = Path(r"C:\Users\hugor\Google Drive\Universidad\Máster en Automática y Robótica\TFM\face recognition\init")
known_folder = Path(r"C:\Users\hugor\Google Drive\Universidad\Máster en Automática y Robótica\TFM\face recognition\known")
unknown_folder = Path(r"C:\Users\hugor\Google Drive\Universidad\Máster en Automática y Robótica\TFM\face recognition\unknown")

input("Press enter.")

# Carga o crea el fichero de datos de caras conocidas y sus codificaciones.

if not os.path.exists('face_database.dat'):
    
    image_gates = face_recognition.load_image_file(init_folder/"Gates.jpg")
    encoding_gates = face_recognition.face_encodings(image_gates)[0]

    image_zuck = face_recognition.load_image_file(init_folder/"Zuckerberg.jpg")
    encoding_zuck = face_recognition.face_encodings(image_zuck)[0]

    image_bezos = face_recognition.load_image_file(init_folder/"Bezos.jpg")
    encoding_bezos = face_recognition.face_encodings(image_bezos)[0]

    image_musk = face_recognition.load_image_file(init_folder/"Elon Musk.jpg")
    encoding_musk = face_recognition.face_encodings(image_musk)[0]
    
    
    known_face_encodings = [encoding_gates, encoding_zuck, encoding_bezos, encoding_musk]
    known_face_names = ["Bill Gates", "Zuckerberg", "Jeff Bezos", "Elon Musk"]
    known_face_times = [time.time(), time.time(), time.time(), time.time()]

    del image_bezos, image_gates, image_musk, image_zuck
    del encoding_bezos, encoding_gates, encoding_musk, encoding_zuck

    with open("face_database.dat", 'wb') as f:
        pickle.dump([known_face_encodings, known_face_names, known_face_times], f)
    print("face_database.dat reiniciado")

else:
    with open("face_database.dat", 'rb') as f:
        known_face_encodings, known_face_names, known_face_times= pickle.load(f)
    print("face_database.dat conservado")


for filename in os.listdir(unknown_folder):
    
    # Carga la imagen en cuestión y localiza las caras.
    test_image = Image.open(unknown_folder/filename)
    test_image.thumbnail((700,700))
    test_image = np.asarray(test_image)
    face_locations = face_recognition.face_locations(test_image)
    face_encodings = face_recognition.face_encodings(test_image, face_locations)
    pil_image = Image.fromarray(test_image)
    draw = ImageDraw.Draw(pil_image)

    # Itera sobre las caras en busca de coinidencias.
    count_unknowns = 0 #cuenta unknowns
    for(top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "unknown face"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        
        draw.rectangle(((left, top),(right, bottom)), outline=(0,0,0))
        text_width, text_height = draw.textsize(name)
        text_width, text_height = text_width, text_height
        draw.rectangle(((left, bottom - text_height - 10),(right, bottom)), fill=(0,0,0), outline=(0,0,0))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255,255,255,255),size=20)

        if name == "unknown face":
            count_unknowns = count_unknowns + 1 # Es True que hay unknowns.

    del draw
    pil_image.show()

    if count_unknowns > 1:
        
        x = input('Conoces algún unknown? (y/n): ')
        
        if x=='y':

            for(top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                
                if not True in matches:
                    
                    face_image = test_image[top-50:bottom+50, left-50:right+50]
                    pil_image = Image.fromarray(face_image)
                    pil_image.show()
                    y = input('Lo conoces? (y/n): ')
                    
                    if y == 'y':
                        
                        new_encoding = face_recognition.face_encodings(face_image)[0]
                        new_name = input('Cuál es su nombre?: ')
                        known_face_encodings.append(new_encoding)
                        known_face_names.append(new_name)
                        known_face_times.append(time.time())

                        with open("face_database.dat", 'wb') as f:
                            pickle.dump([known_face_encodings, known_face_names, known_face_times], f)

                        im = pil_image.convert("RGB")
                        im.save('known/' + new_name + '.jpg',"jpeg")

                        print("face_database.dat actualizado")
                        
    elif (count_unknowns == 1) & (len(face_locations) == 1):
        
        for(top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                
            if not True in matches:
                
                face_image = test_image[top-50:bottom+50, left-50:right+50]
                y = input('Lo conoces? (y/n): ')
                
                if y == 'y':
                    
                    new_encoding = face_recognition.face_encodings(face_image)[0]
                    new_name = input('Cuál es su nombre?: ')
                    known_face_encodings.append(new_encoding)
                    known_face_names.append(new_name)
                    known_face_times.append(time.time())

                    with open("face_database.dat", 'wb') as f:
                        pickle.dump([known_face_encodings, known_face_names, known_face_times], f)

                    im = pil_image.convert("RGB")
                    im.save('known/' + new_name + '.jpg',"jpeg")

                    print("face_database.dat actualizado")
                    
    elif (count_unknowns == 1) & (len(face_locations) > 1):
        
        for(top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                
            if not True in matches:
                
                face_image = test_image[top-50:bottom+50, left-50:right+50]
                y = input('Conoces al desconocido de la imagen? (y/n): ')
                
                if y == 'y':
                    
                    new_encoding = face_recognition.face_encodings(face_image)[0]
                    new_name = input('Cuál es su nombre?: ')
                    known_face_encodings.append(new_encoding)
                    known_face_names.append(new_name)
                    known_face_times.append(time.time())

                    with open("face_database.dat", 'wb') as f:
                        pickle.dump([known_face_encodings, known_face_names, known_face_times], f)

                    im = pil_image.convert("RGB")
                    im.save('known/' + new_name + '.jpg',"jpeg")

                    print("face_database.dat actualizado")

for i in range(len(known_face_encodings)):
    print(known_face_names[i],known_face_times[i])

print(len(known_face_encodings),len(known_face_names),len(known_face_times))