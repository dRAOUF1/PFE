import cv2
from Detector import Detector

detector = Detector("models/yolov8n-face.pt")

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("C:/Users/yas/Downloads/djalilcas.mov")
while True:
    ret, frame = cap.read()
    aligned = None
    if not ret:
        break
    Faces = detector.detect(frame)
    if Faces is not None:
        frame = detector.Draw(frame, Faces)
    if len(Faces) > 0:
        aligned = detector.align1(frame, Faces)
    if aligned is not None:
        for align in aligned:
            cv2.imshow(f"aligned", align)

    cv2.imshow('frame', frame)
    k = cv2.waitKey(10)
    if k == 27:         # wait for ESC key to exit
        break
cap.release()

# frame = cv2.imread("C:/Users/yas/Desktop/Proj/depface/db/djalil4.jpg")
# # frame = cv2.resize(frame,(480,640))
# # cv2.imshow('test',frame)
# # cv2.waitKey(0)
# faces = detector.detect(frame)
# aligned = detector.align1(frame,faces)
# cv2.imshow("original",frame)
# cv2.imshow('aligned',aligned[0])

# cv2.waitKey(0)
cv2.destroyAllWindows()