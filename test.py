import numpy as np
from Face import Face
from App import App
import cv2


app = App()
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    faces = app.extractFaces(frame)
    for face in faces:
        frame = face.align_and_rotate(frame)
        frame = app.Draw(frame, face,keypoints=True)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break