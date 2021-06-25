import face_recognition
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import pickle
import os
import time

os.system("cls") # Limpia command line.

# Localiza las carpetas de fotos.
unknown_folder = Path(r"C:\Users\hugor\Google Drive\Universidad\Máster en Automática y Robótica\TFM\face recognition\experimento 1")

count = 0
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
        
        draw.rectangle(((left, top),(right, bottom)), outline=(0,0,0))
        text_width, text_height = draw.textsize('Cara')
        text_width, text_height = text_width, text_height
        draw.rectangle(((left, bottom - text_height - 10),(right, bottom)), fill=(0,0,0), outline=(0,0,0))
        draw.text((left + 6, bottom - text_height - 5), 'Cara', fill=(255,255,255,255),size=20)

        count = count + 1

    del draw
    pil_image.show()

end = time.time()

print(count, start, end, (end-start)/count)