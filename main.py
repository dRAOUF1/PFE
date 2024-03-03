from App import App
import cv2 ,time

from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image

# Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=True)

# Load pre-trained FaceNet model
resnet = InceptionResnetV1(pretrained='casia-webface').eval()
cap = cv2.VideoCapture(0)
while True:
    # Load an image containing faces
    ret,img = cap.read()
    # Detect faces in the image
    boxes, _ = mtcnn.detect(img)

    # If faces are detected, extract embeddings
    if boxes is not None:
        aligned = mtcnn(img)
        embeddings = resnet(aligned).detach()
    #draw boxes
    for box in boxes:
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #make a request with face name
    
