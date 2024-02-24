from ultralytics import YOLO
import cv2
from Face import Face

class Detector():
    def __init__(self, model_path):
        self.model = YOLO(model_path,task='detect')

    def detect(self, frame):
        faces = []
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model(frame,verbose=False)
        try:
            for result in results:
                boxes = result.boxes
                confidence = result.boxes.conf.tolist()[0]
                x1, y1, x2, y2= int(boxes.xyxy[0][0]),int(boxes.xyxy[0][1]),int(boxes.xyxy[0][2]),int(boxes.xyxy[0][3])
                if confidence > 0.7:# and (x2-x1) > 30 and (y2-y1) > 30:

                    left_eye = result.keypoints.xy[0][0].tolist()
                    right_eye = result.keypoints.xy[0][1].tolist()
                    nose = result.keypoints.xy[0][2].tolist()
                    left_mouth = result.keypoints.xy[0][3].tolist()
                    right_mouth = result.keypoints.xy[0][4].tolist()
                    #convert to tuples of int
                    left_eye = tuple(int(i) for i in left_eye)
                    right_eye = tuple(int(i) for i in right_eye)
                    nose = tuple(int(i) for i in nose)
                    left_mouth = tuple(int(i) for i in left_mouth)
                    right_mouth = tuple(int(i) for i in right_mouth)                
                    
                    face = Face(x=x1,y=y1,x2=x2,y2=y2,confidence=confidence,left_eye=left_eye,right_eye=right_eye,nose=nose,left_mouth=left_mouth,right_mouth=right_mouth)
                    faces.append(face)

            return faces
        except(IndexError):
            return None
    def Draw(self, frame, faces):
        for face in faces:
            x1, y1, x2, y2 = face.x, face.y, face.x2, face.y2
            left_eye = face.left_eye
            right_eye = face.right_eye
            nose = face.nose
            left_mouth = face.left_mouth
            right_mouth = face.right_mouth
            confidence = face.confidence
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            frame = cv2.putText(frame, f"{confidence:.3f}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            frame = cv2.circle(frame, left_eye, 5, (0, 0, 255), 2)
            frame = cv2.circle(frame, right_eye, 5, (0, 0, 255), 2)
            frame = cv2.circle(frame, nose, 5, (255, 0, 0), 2)
            frame = cv2.circle(frame, left_mouth, 5, (0, 255, 255), 2)
            frame = cv2.circle(frame, right_mouth, 5, (0, 255, 255), 2)
        return frame
    def extract(self, frame,faces:Face):
        # test
        extracted = []
        for face in faces:
            extracted.append(frame[face.y1:face.y2, face.x1:face.x2])
        return extracted