import cv2
import numpy as np
import joblib



class Spoofing:
    def __init__(self,model_path="C:/Users/yas/Desktop/PFE/models/spoof.pkl"):
        self.model = joblib.load(model_path)

    def preprocess_image(self, image):
        img_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
        img_luv = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)

        ycrcb_hist = self.calc_hist(img_ycrcb)
        luv_hist = self.calc_hist(img_luv)

        feature_vector = np.append(ycrcb_hist.ravel(), luv_hist.ravel())
        feature_vector = feature_vector.reshape(1, len(feature_vector))
        return feature_vector

    def calc_hist(self,img):
        histogram = [0] * 3
        for j in range(3):
            histr = cv2.calcHist([img], [j], None, [256], [0, 256])
            histr *= 255.0 / histr.max()
            histogram[j] = histr
        return np.array(histogram)
    
    def is_spoof(self, image):
        # Prétraiter l'image
        feature_vector = self.preprocess_image(image)

        # Faire une prédiction avec le modèle
        prediction = self.model.predict_proba(feature_vector)

        # Renvoyer la prédiction
        return prediction
