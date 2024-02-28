from Detector import Detector
from Recogniser import Recogniser
import os,cv2


class App:
    def __init__(self,db_path,recognition_model_path="models/face_recognition_sface_2021dec.onnx",detection_model_path='models/face_detection_yunet_2023mar.onnx'):
        self.detector = Detector(modelPath=detection_model_path,
                     inputSize=[320, 320],
                     confThreshold=0.65,
                     nmsThreshold=0.3,
                )
        self.recognizer = Recogniser(modelPath=recognition_model_path, disType=0)
        self.embeddings = self._dbToEmbeddings(db_path)

    
    def _dbToEmbeddings(self,db):
        images = [os.path.join(dossier, fichier) for dossier, sous_dossiers, fichiers in os.walk(db) for fichier in fichiers if fichier.endswith('.jpg') or fichier.endswith('.png') or fichier.endswith('.jpeg')]
        embeddings = {im.split('/')[-1].split('\\')[-1].split('.')[0]:None for im in images}
        for image in images:
            img = cv2.imread(image)
            self.detector.setInputSize([img.shape[1], img.shape[0]])
            faces = self.detector.infer(img)
            if len(faces) > 0:
                for face in faces:
                    embedding = self.recognizer.infer(img,face.toArray()[:-1])
                    embeddings[image.split('/')[-1].split('\\')[-1].split('.')[0]] = embedding
        return embeddings
    
    def extractFaces(self,frame):
        self.detector.setInputSize([frame.shape[1], frame.shape[0]])
        faces = self.detector.infer(frame)
        return faces
    
    def find_match(self,image):
        minDist = 0
        minKey = None
        recognizedFaces = []
        self.detector.setInputSize([image.shape[1], image.shape[0]])
        faces = self.detector.infer(image)
        if len(faces) > 0:
            for face in faces:
                embedding = self.recognizer.infer(image,face.toArray()[:-1])
                for key, value in self.embeddings.items():
                    if value is not None:
                        result = self.recognizer.match(embedding, value)
                        if result != 0:
                            if minDist == 0:
                                minDist = result
                                minKey = key
                                face.name = minKey
                            elif result > minDist:
                                minDist = result
                                minKey = key
                                face.name = minKey
                if face.name != "":
                    recognizedFaces.append(face)
        return minDist,recognizedFaces
    
    
    def Draw(self, frame, face,keypoints = False):
            
        x1, y1, x2, y2 = int(face.x1), int(face.y1), int(face.x2), int(face.y2)
        confidence = face.confidence
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        frame = cv2.putText(frame, f"{confidence:.3f}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        frame = cv2.rectangle(frame, (x1,y2), (x2,y2+15), (255, 0, 0) , -1)
        
        cv2.putText(frame, face.name + f"" , (x1,y2+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA) 
        if keypoints:
            left_eye = (int(face.left_eye[0]),int(face.left_eye[1]))
            right_eye = (int(face.right_eye[0]),int(face.right_eye[1]))
            nose = (int(face.nose[0]),int(face.nose[1]))
            left_mouth = (int(face.left_mouth[0]),int(face.left_mouth[1]))
            right_mouth = (int(face.right_mouth[0]),int(face.right_mouth[1]))
            frame = cv2.circle(frame, left_eye, 3, (0, 0, 255), 2)
            frame = cv2.circle(frame, right_eye, 3, (0, 0, 255), 2)
            frame = cv2.circle(frame, nose, 3, (255, 0, 0), 2)
            frame = cv2.circle(frame, left_mouth, 3, (0, 255, 255), 2)
            frame = cv2.circle(frame, right_mouth, 3, (0, 255, 255), 2)
        return frame