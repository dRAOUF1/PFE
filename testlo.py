from VideoStream import VideoStream
from App import App
import cv2
from centroidtracker2 import CentroidTracker
import pickle
import numpy as np

def func(em1,em2):
    if em1 is None or em2 is None:
        return None
    try:
        vecteur_numpy = np.asarray(em1)
        vecteur_numpy2 = np.asarray(em2).T
        
        vecteur_2d = np.dot(vecteur_numpy,vecteur_numpy2)/(np.linalg.norm(vecteur_numpy)*np.linalg.norm(vecteur_numpy2))
        return vecteur_2d
    except Exception as e:
        print(f"Erreur lors de la conversion du vecteur numpy: {str(e)}")
        return None

app = App("http://localhost:3001/getEmbeddings/all/all")
ref = cv2.imread("C:/Users/yas/Desktop/tempsdb/ayoub.jpg")
nref = cv2.imread("C:/Users/yas/Desktop/tempsdb/djalil.jpg")
em1 = app.get_embedding(ref)
# nem1 = app.get_embedding(nref)
nem1 = app.embeddings["202031044688"]

cap = VideoStream("C:/Users/yas/Downloads/Archive (3)/")
cap.start()
while True:
    frame = cap.read()
    if frame is None:
        break
    #frame = cv2.resize(frame,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    em2 = app.get_embedding(frame)
    em = func(em1,em2)
    nem = func(nem1,em2)
    # print("em1: ",em,"nem1: ",nem)
    if nem is not None and float(nem[0][0]) >0.37:
        print("djalil",nem)
        cv2.waitKey(0)


    cv2.imshow("frame",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.stop()
