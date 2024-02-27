
import sys
import argparse

import numpy as np
import cv2 ,os

from Recogniser import Recogniser
from Detector import Detector

def dbToEmbeddings(db, recognizer, detector):
    images = [os.path.join(dossier, fichier) for dossier, sous_dossiers, fichiers in os.walk(db) for fichier in fichiers if fichier.endswith('.jpg') or fichier.endswith('.png') or fichier.endswith('.jpeg')]
    embeddings = {im.split('/')[-1].split('.')[0]:None for im in images}
    for image in images:
        img = cv2.imread(image)
        detector.setInputSize([img.shape[1], img.shape[0]])
        face = detector.infer(img)
        if len(face) > 0:
            embedding = recognizer.infer(img,face[0].toArray()[:-1])
            embeddings[image.split('/')[-1].split('.')[0]] = embedding
    return embeddings

def find_match(image, embeddings, recognizer, detector):
    minDist = 0
    minKey = None
    img = image
    detector.setInputSize([img.shape[1], img.shape[0]])
    face = detector.infer(img)
    if len(face) > 0:
        embedding = recognizer.infer(img,face[0].toArray()[:-1])
        for key, value in embeddings.items():
            if value is not None:
                result = recognizer.match(embedding, value)
                if result != 0:
                    if minDist == 0:
                        minDist = result
                        minKey = key
                    elif result > minDist:
                        minDist = result
                        minKey = key

    return minKey,minDist
    
def find_match2(img1,db,recognizer,detector):
    images = [os.path.join(dossier, fichier) for dossier, sous_dossiers, fichiers in os.walk(db) for fichier in fichiers if fichier.endswith('.jpg') or fichier.endswith('.png') or fichier.endswith('.jpeg')]

    detector.setInputSize([img1.shape[1], img1.shape[0]])
    face1 = detector.infer(img1)
    if len(face1)>0:
        face1=face1[0].toArray()
        for image in images:
            # try:
                img = cv2.imread(image)
                detector.setInputSize([img.shape[1], img.shape[0]])
                face2 = detector.infer(img)
                if len(face2)>0:
                    face2=face2[0].toArray()
                result = recognizer.match(img1, face1[:-1], img, face2[:-1])
                if result >= 0.363:
                    print(image.split('/')[-1], result)
            # except(IndexError):
            #     continue

if __name__ == '__main__':

    detector = Detector(modelPath='models/face_detection_yunet_2023mar.onnx',
                     inputSize=[320, 320],
                     confThreshold=0.7,
                     nmsThreshold=0.3,
                )
    recognizer = Recogniser(modelPath="models/face_recognition_sface_2021dec.onnx", disType=0)
    embeddings = dbToEmbeddings("C:/Users/yas/Desktop/tempyolov8/",recognizer, detector)

    cap = cv2.VideoCapture("C:/Users/yas/Downloads/djalilcap.mov")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("lol",frame)
        pers,dist = find_match(frame,embeddings,recognizer,detector)
        if pers is not None:
            print(pers, dist)
        else:
            print('unknown')
        
        k = cv2.waitKey(10)
        if k == 27:         # wait for ESC key to exit
            break
        
        
    # Instantiate SFace for face recognition
    # # Instantiate YuNet for face detection

    # img1 = cv2.imread("./zouj.jpg")
    # # img2 = cv2.imread(args.input2)

    # # Detect faces
    # detector.setInputSize([img1.shape[1], img1.shape[0]])
    # face1 = detector.infer(img1)
    # recognizer.infer(img1,face1[0].toArray()[:-1])













    # assert len(face1) > 0, 'Cannot find a face in {}'.format("C:/Users/yas/Desktop/tempyolov8/amina.jpg")
    # cap = cv2.VideoCapture("C:/Users/yas/Downloads/djalil.mov")
    # while True:
    #     ret, img2 = cap.read()
    #     if not ret:
    #         break
    #     detector.setInputSize([img2.shape[1], img2.shape[0]])
    #     face2 = detector.infer(img2)
    #     if len(face2)> 0:
    #         img2 = detector.Draw(img2, face2)
    #     cv2.imshow("camera input", img2)
    #     # if not face2.shape[0] > 0:
    #     #     print('Cannot find a face in {}'.format("camera input"))

    #     # Match
    #     # try:
    #     #     result = recognizer.match(img1, face1[0][:-1], img2, face2[0][:-1])
    #     #     print('Result: {}.'.format('same identity' if result else 'different identities'))
    #     #     if result:
    #     #         print('Confidence: {:.2f}.'.format(recognizer.confidence))
    #     #         break
    #     # except (IndexError):
    #     #     print('Cannot find a face in {}'.format("camera input"))

        
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
