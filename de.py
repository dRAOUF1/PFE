from App import App
from VideoStream import VideoStream
from Spoofing import Spoofing
import cv2
import numpy as np
import time

app = App("http://localhost:3001/getEmbeddings/all/all")
# app.facesFromVideo("E:/db/P1E_S1_C1.avi","E:/db/faces")

app.embeddings = app.localDbToEmbeddings("E:/db/faces")
spoof = Spoofing()
classes = spoof.model.classes_
print(len(app.embeddings))
cap = VideoStream(0)
cap.start()
while True:
    frame = cap.read()
    if frame is None:
        break
    # frame = cv2.rotate(frame, cv2.ROTATE_180)
    faces = app.extractFaces(frame)
    for face in faces:
        #detect spoof
        face_img = frame[int(face.y1):int(face.y2),int(face.x1):int(face.x2)]
        # time_start = time.time()
        prediction = spoof.is_spoof(face_img)
        # print("Time taken for prediction: ",time.time()-time_start)
        

        # Afficher les probabilités prédites avec les classes correspondantes
        # for i, probs in enumerate(prediction):
        #     for j, prob in enumerate(probs):
        #         print(f"Classe {classes[j]}: {prob:.4f}")
        # print(prediction)
        if 0.1 < prediction[0][0]:
            pass
            # print("Real face",prediction[0][0])
        else:
            pass
            print("Spoof detected",prediction[0][1])
    cv2.imshow("frame",frame)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.stop()
print("fin")