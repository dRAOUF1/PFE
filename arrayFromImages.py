import os,cv2
from pymongo import MongoClient
        


if __name__ == "__main__":
    DB = "C:/Users/yas/Desktop/tempsdb"
    client = MongoClient("mongodb://localhost:27017/")
    db = client["mydb"]
    # print(client.list_database_names())
    # exit(0)
    detector = cv2.FaceDetectorYN.create(
            model="models/face_detection_yunet_2023mar.onnx",
            config="",
            input_size=[320,320],
            score_threshold=0.65,
            nms_threshold=0.3,
            top_k=5000,
            backend_id=0,
            target_id=0)
    recognizer = cv2.FaceRecognizerSF.create(
            model="models/face_recognition_sface_2021dec.onnx",
            config="",
            backend_id=0,
            target_id=0)

    
    images = [os.path.join(dossier, fichier) for dossier, sous_dossiers, fichiers in os.walk(DB) for fichier in fichiers if fichier.endswith('.jpg') or fichier.endswith('.png') or fichier.endswith('.jpeg')]
        #mbeddings = {im.split('/')[-1].split('\\')[-1].split('.')[0]:None for im in images}
    for image in images:
        img = cv2.imread(image)
        detector.setInputSize([img.shape[1], img.shape[0]])
        faces = detector.detect(img)
        if len(faces) > 0:
            inputBlob = recognizer.alignCrop(img,faces[1][:-1])
            embedding = recognizer.feature(inputBlob)
            etudiant = {"MatriculeEtd":image.split('/')[-1].split('\\')[-1].split('.')[0], "embedding":embedding.tolist()}
            result=db.embeddings.insert_one(etudiant)
            # print(image.split('/')[-1].split('\\')[-1].split('.')[0], embedding)