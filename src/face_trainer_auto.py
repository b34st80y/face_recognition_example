import face_recognition
import os
import pickle

print("Training...")

face_encodings = {}
image_dir = "../images"

MODEL = "hog"

# Traverse the image_dir directory and encode each face
for root, dirs, files, in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("PNG") or file.endswith("jpg") or file.endswith("JPEG"):
            path = os.path.join(root, file)
            fileName = os.path.basename(path).replace(" ", "_").lower()
            arr = fileName.split(".")
            label = arr[0]
            image = face_recognition.load_image_file(path)
            print(label, path)
            if label not in face_encodings:
                face_encodings[label] = face_recognition.face_encodings(image, model=MODEL)[0]

with open('faces_model.yaml', 'wb') as f:
    pickle.dump(face_encodings, f)

print("Done!")
