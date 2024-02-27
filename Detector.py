import numpy as np
import cv2
from Face import Face

class Detector:
    def __init__(self, modelPath, inputSize=[320, 320], confThreshold=0.6, nmsThreshold=0.3):
        self._modelPath = modelPath
        self._inputSize = tuple(inputSize) # [w, h]
        self._confThreshold = confThreshold
        self._nmsThreshold = nmsThreshold
        # self._topK = topK
        # self._backendId = backendId
        # self._targetId = targetId

        self._model = cv2.FaceDetectorYN.create(
            model=self._modelPath,
            config="",
            input_size=self._inputSize,
            score_threshold=self._confThreshold,
            nms_threshold=self._nmsThreshold,
            top_k=5000,
            backend_id=0,
            target_id=0)


    def setInputSize(self, input_size):
        self._model.setInputSize(tuple(input_size))

    def infer(self, image):
        # Forward
        results = self._model.detect(image)
        if results[1] is not None:
            faces = [ Face(
                x=face[0],
                y=face[1],
                x2=face[0] + face[2],
                y2=face[1] + face[3],
                confidence=face[14],
                right_eye=(face[4], face[5]),
                left_eye=(face[6], face[7]),
                nose=(face[8], face[9]),
                right_mouth=(face[10], face[11]), # (x, y)
                left_mouth=(face[12], face[13]) # (x, y
            ) for face in results[1]]
            return faces
        
        return []

    def Draw(self, frame, faces):
            for face in faces:
                x1, y1, x2, y2 = int(face.x1), int(face.y1), int(face.x2), int(face.y2)
                left_eye = int(face.left_eye)
                right_eye = int(face.right_eye)
                nose = int(face.nose)
                left_mouth = int(face.left_mouth)
                right_mouth = int(face.right_mouth)
                confidence = face.confidence
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                frame = cv2.putText(frame, f"{confidence:.3f}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                frame = cv2.circle(frame, left_eye, 1, (0, 0, 255), 1)
                frame = cv2.circle(frame, right_eye, 1, (0, 0, 255), 1)
                frame = cv2.circle(frame, nose, 1, (255, 0, 0), 1)
                frame = cv2.circle(frame, left_mouth, 1, (0, 255, 255), 1)
                frame = cv2.circle(frame, right_mouth, 1, (0, 255, 255), 1)
            return frame