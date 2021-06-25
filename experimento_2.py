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
unknown_folder = Path(r"C:\Users\hugor\Google Drive\Universidad\Máster en Automática y Robótica\TFM\face recognition\experimento 2")

# Carga o crea el fichero de datos de caras conocidas y sus codificaciones.

if not os.path.exists('face_database_2.dat'):
    
    image_gates = face_recognition.load_image_file(init_folder/"Gates.jpg")
    encoding_gates = face_recognition.face_encodings(image_gates)[0]

    image_zuck = face_recognition.load_image_file(init_folder/"Zuckerberg.jpg")
    encoding_zuck = face_recognition.face_encodings(image_zuck)[0]

    image_bezos = face_recognition.load_image_file(init_folder/"Bezos.jpg")
    encoding_bezos = face_recognition.face_encodings(image_bezos)[0]

    image_musk = face_recognition.load_image_file(init_folder/"Elon Musk.jpg")
    encoding_musk = face_recognition.face_encodings(image_musk)[0]
    
    image_brad = face_recognition.load_image_file(init_folder/"Brad.jpg")
    encoding_brad = face_recognition.face_encodings(image_brad)[0]
    
    known_face_encodings = [encoding_gates, encoding_zuck, encoding_bezos, encoding_musk, encoding_brad]
    known_face_names = ["Bill Gates", "Zuckerberg", "Jeff Bezos", "Elon Musk", "Brad Pitt"]
    known_face_times = [time.time(), time.time(), time.time(), time.time(), time.time()]

    del image_bezos, image_gates, image_musk, image_zuck, image_brad
    del encoding_bezos, encoding_gates, encoding_musk, encoding_zuck, encoding_brad

    with open("face_database_2.dat", 'wb') as f:
        pickle.dump([known_face_encodings, known_face_names, known_face_times], f)
    print("face_database_2.dat reiniciado")

else:
    with open("face_database_2.dat", 'rb') as f:
        known_face_encodings, known_face_names, known_face_times= pickle.load(f)
    print("face_database_2.dat conservado")


count = 0
count_unknowns = 0
start = time.time()

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
    
    for(top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "No"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        
        draw.rectangle(((left, top),(right, bottom)), outline=(0,0,0))
        text_width, text_height = draw.textsize(name)
        text_width, text_height = text_width, text_height
        draw.rectangle(((left, bottom - text_height - 10),(right, bottom)), fill=(0,0,0), outline=(0,0,0))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255,255,255,255),size=20)

        if name == "Brad Pitt":
            count = count + 1

        if name == "No":
            count_unknowns= count_unknowns + 1

    del draw
    pil_image.show()


end = time.time()

print(count, start, end, (end-start)/count)
print(count_unknowns)