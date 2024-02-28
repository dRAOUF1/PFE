import csv,time,cv2
from APP2 import App2


import os

metrics = [0,1]
tests = ["C:/Users/yas/Downloads/ayoub.mov"]
db = "/kaggle/input/base-db"

app = App2("C:/Users/yas/Desktop/tempsdb")  
cap = cv2.VideoCapture(tests[0])

while True:
    ret, frame = cap.read()
    if not ret:
        break
    dist,faces = app.find_match(frame)
    if len(faces)>0:
        for face in faces:
            frame = app.Draw(frame,face)
    cv2.imshow("frame",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
