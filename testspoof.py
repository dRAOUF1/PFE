import numpy as np
import tensorflow as tf
import cv2
from App import App

def preprocess_image(image):
        img_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
        img_luv = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)

        ycrcb_hist = calc_hist(img_ycrcb)
        luv_hist = calc_hist(img_luv)

        feature_vector = np.append(ycrcb_hist.ravel(), luv_hist.ravel())
        feature_vector = feature_vector.reshape(1, len(feature_vector))
        return feature_vector

def calc_hist(img):
    histogram = [0] * 3
    for j in range(3):
        histr = cv2.calcHist([img], [j], None, [256], [0, 256])
        histr *= 255.0 / histr.max()
        histogram[j] = histr
    return np.array(histogram)

# Charger le modèle depuis le fichier .h5
model = tf.keras.models.load_model('models/clf2.h5')
#summary
# model.summary()

app = App("http://localhost:3001/getEmbeddings/all/all")
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # frame = cv2.rotate(frame, cv2.ROTATE_180)
    faces = app.extractFaces(frame)
    # print(len(faces))
    for face in faces:
        # try:
        face_img = frame[int(face.y1):int(face.y2),int(face.x1):int(face.x2)]
        hist = preprocess_image(face_img)
        # Supposons que 'probas' contient les probabilités prédites par votre modèle pour un ensemble d'échantillons
        probas = model.predict(hist,verbose=0)
        # print(probas)

        # Obtenir les classes prédites pour chaque échantillon
        # predicted_classes = np.argmax(probas, axis=1)

        # # # Supposons que 'class_names' contient la liste des noms de classe correspondant aux indices de classe
        # class_names = ["spoof",'live']  # Remplacez [...] par votre liste de noms de classe

        # # # Obtenir les noms de classe prédits pour chaque échantillon
        # predicted_class_names = [class_names[i] for i in predicted_classes]
        # print(predicted_class_names)
        if probas[0][0] > 0.85:
                print("reel face",probas[0])
        # except:
        #     continue
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Prétraiter vos données si nécessaire (par exemple, redimensionner, normaliser)

# Effectuer des prédictions sur de nouvelles données
# predictions = model.predict(vos_donnees)

# Afficher les prédictions
# print(predictions)
