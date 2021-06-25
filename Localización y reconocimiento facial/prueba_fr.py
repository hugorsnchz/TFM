import face_recognition
from pathlib import Path
from PIL import Image, ImageDraw

groups_folder = Path(r"C:\Users\hugor\Google Drive\Universidad\Máster en Automática y Robótica\TFM\Proyecto\groups")
known_folder = Path(r"C:\Users\hugor\Google Drive\Universidad\Máster en Automática y Robótica\TFM\Proyecto\known")
unknown_folder = Path(r"C:\Users\hugor\Google Drive\Universidad\Máster en Automática y Robótica\TFM\Proyecto\unknown")

##########################################################################################
# Codifica caras, las reconoce y luego pinta un cuadrado con una etiqueta ################

# Encondig faces
image_gates = face_recognition.load_image_file(known_folder/"Bil Gates.jpg")
gates_encoding = face_recognition.face_encodings(image_gates)[0]

image_elon = face_recognition.load_image_file(known_folder/"elon.jpg")
elon_encoding = face_recognition.face_encodings(image_elon)[0]

known_face_encodings = [
    gates_encoding,
    elon_encoding
]

known_face_names = [
    "Bill Gates",
    "Elon Musk"
]

# Load test image and find faces
test_image = face_recognition.load_image_file(groups_folder/"gatesfamily.jpg")
face_locations = face_recognition.face_locations(test_image)
face_encodings = face_recognition.face_encodings(test_image, face_locations)
pil_image = Image.fromarray(test_image)
draw = ImageDraw.Draw(pil_image)

# Loop over faces in test_image
for(top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

    name = "unknown face"

    if True in matches:
        first_match_index = matches.index(True)
        name = known_face_names[first_match_index]
    
    draw.rectangle(((left, top),(right, bottom)), outline=(0,0,0))
    text_width, text_height = draw.textsize(name)
    draw.rectangle(((left, bottom - text_height - 10),(right, bottom)), fill=(0,0,0), outline=(0,0,0))
    draw.text((left + 6, bottom - text_height - 5), name, fill=(255,255,255,255))

del draw

pil_image.show()

##########################################################################################
# Codifica caras y luego las compara #####################################################

# Codificación
image_gates = face_recognition.load_image_file(known_folder/"gates.jpg")
gates_encoding = face_recognition.face_encodings(image_gates)[0]

image_unknown = face_recognition.load_image_file(unknown_folder/"elon2.jpg")
unknown_encoding = face_recognition.face_encodings(image_unknown)[0]

# Comparing encoded faces
results = face_recognition.compare_faces([gates_encoding], unknown_encoding)
print(results)

##########################################################################################
# Busca caras en una imagen y las cuenta #################################################

# Searching for faces in an image
test_image = face_recognition.load_image_file(groups_folder/"gatesfamily.jpg")
face_locations = face_recognition.face_locations(test_image)

# Iterating over the faces found
for face_location in face_locations:
    top, right, bottom, left = face_location
    face_image = test_image[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
    #pil_image.show()   # enseña las imágenes
    #pil_image.save(f'{top}.jpg')   #guarda las imágenes con uno de los números del cuadrado
