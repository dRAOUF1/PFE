from App import App
import cv2, joblib
import numpy as np

def get_match(model,embedding_dict,vector):
    max_proba = 0
    max_name = ""
    for name, embedding in embedding_dict.items():
        feature_vector = np.append(embedding, vector)
        feature_vector = feature_vector.reshape(1, len(feature_vector))
        proba = model.predict_proba(feature_vector)
        # prediction = model.predict(feature_vector)
        if proba[0][1] > max_proba:
            max_proba = proba[0][1]
            max_name = name
    if max_proba > 0.91:
        return max_name
    
app = App("http://localhost:3001/getEmbeddings/all/all")
model = joblib.load("C:/Users/yas/Desktop/PFE/models/dist.pkl")
count = 0
cap = cv2.VideoCapture("C:/Users/yas/Desktop/test.avi")
while True:
    ret,frame = cap.read()
    if frame is None:
        break
    count += 1
    if count%2 == 0:
        continue

    frame = cv2.rotate(frame, cv2.ROTATE_180)
    faces = app.extractFaces(frame)
    for face in faces:
        #calculer le flou de l'image
        face_img = frame[int(face.y1):int(face.y2),int(face.x1):int(face.x2)]
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        fm = cv2.Laplacian(gray, cv2.CV_64F).var()
        if fm < 30:
            continue
        embedding = app.get_embedding(frame,face)
        if embedding is None:
            continue
        name = get_match(model,app.embeddings,embedding)
        if name is None:
            continue
        print(name)
        # frame = app.Draw(frame,face,name)
    cv2.imshow("frame",frame)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break








# i1 = cv2.imread("C:/Users/yas/Desktop/tempsdb/kie.jpg")
# i2 = cv2.imread("C:/Users/yas/Desktop/tempsdb/ayoub.jpg")
# f1 = app.get_embedding(i1)
# f2 = app.get_embedding(i2)
# # print("f1",f1,"f2",f2)
# model = joblib.load("C:/Users/yas/Desktop/PFE/models/dist.pkl")
# feature_vector = np.append(f1, f2)
# feature_vector = feature_vector.reshape(1, len(feature_vector))
# prediction = model.predict(feature_vector)
# print(prediction)