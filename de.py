from App import App
# from VideoStream import VideoStream
from Spoofing import Spoofing
import cv2
import numpy as np
import time

app = App("http://localhost:3001/getEmbeddings/all/all")
# app.facesFromVideo("E:/db/P1E_S1_C1.avi","E:/db/faces")

app.embeddings = app.localDbToEmbeddings("E:/db/faces")
spoof = Spoofing("models/spoof_Random Forest new.pkl")
classes = spoof.model.classes_
print(len(app.embeddings))
cap = cv2.VideoCapture("E:/db/P1E_S1_C1.avi")
minpred =10
maxpred = 0
# cap.start()
while True:
    ret,frame = cap.read()
    if frame is None:
        break
    # frame = cv2.rotate(frame, cv2.ROTATE_180)
    faces = app.extractFaces(frame)
    for face in faces:
        #detect spoof
        face_img = frame[int(face.y1)-50:int(face.y2)+50,int(face.x1)-20:int(face.x2)+20]
        # time_start = time.time()
        prediction = spoof.is_spoof(face_img)
        if prediction is None:
            continue
        print(prediction)
        # print("Time taken for prediction: ",time.time()-time_start)

        # Afficher les probabilités prédites avec les classes correspondantes
        # for i, probs in enumerate(prediction):
        #     for j, prob in enumerate(probs):
        #         print(f"Classe {classes[j]}: {prob:.4f}")
        # print(prediction)
        if minpred > prediction[0][1]:
            minpred = prediction[0][1]
        elif maxpred < prediction[0][1]:
            maxpred = prediction[0][1]
        if prediction[0][1] > 0.75: #0.67 < prediction[0][1]:
            label = f"Real face"# {prediction[0][1]}"
            color = (0, 255, 0)  # Green color for real face
        else:
            label = f"Spoof face"# {prediction[0][0]}"
            color = (0, 0, 255)  # Red color for spoof face

        # # Dessiner un rectangle autour du visage
        cv2.rectangle(frame, (int(face.x1), int(face.y1)), (int(face.x2), int(face.y2)), color, 2)
        # # # Dessiner le label
        cv2.putText(frame, label, (int(face.x1), int(face.y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        # app.Draw(frame,face)
        # print(face.name,face.distance)
    cv2.imshow("frame",frame)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

# cap.stop()
print("fin")
print("min",minpred,"max",maxpred)