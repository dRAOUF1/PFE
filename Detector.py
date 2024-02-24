from ultralytics import YOLO
import cv2
from Face import Face

class Detector():
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, frame):
        faces = []
        results = self.model(frame)
        try:
            for result in results:
                boxes = result.boxes
                x1, y1, x2, y2= int(boxes.xyxy[0][0]),int(boxes.xyxy[0][1]),int(boxes.xyxy[0][2]),int(boxes.xyxy[0][3])

                left_eye = result.keypoints.xy[0][0].tolist()
                right_eye = result.keypoints.xy[0][1].tolist()

                #convert to tuples of int
                left_eye = tuple(int(i) for i in left_eye)
                right_eye = tuple(int(i) for i in right_eye)
            
                confidence = result.boxes.conf.tolist()[0]
                
                face = Face(x=x1,y=y1,x2=x2,y2=y2,confidence=confidence,left_eye=left_eye,right_eye=right_eye)
                faces.append(face)

            return faces
        except(IndexError):
            return None
    def Draw(self, frame, faces):
        for face in faces:
            x1, y1, x2, y2 = face.x, face.y, face.x2, face.y2
            left_eye = face.left_eye
            right_eye = face.right_eye
            confidence = face.confidence
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            frame = cv2.putText(frame, str(confidence), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            frame = cv2.circle(frame, left_eye, 5, (0, 0, 255), 2)
            frame = cv2.circle(frame, right_eye, 5, (0, 0, 255), 2)
        return frame
    def align(self, frame, x1, y1, x2, y2):
        return frame[y1:y2, x1:x2]