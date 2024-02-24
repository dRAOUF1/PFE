import cv2
from Detector import Detector

detector = Detector("models\yolov8n-face.pt")

cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("C:/Users/yas/Downloads/djalilcas.mov")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    Faces = detector.detect(frame)
    if Faces is not None:
        frame = detector.Draw(frame, Faces)
    cv2.imshow('frame', frame)
    k = cv2.waitKey(10)
    if k == 27:         # wait for ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()