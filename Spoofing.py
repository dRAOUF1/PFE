import cv2
import numpy as np
import joblib
from skimage import feature


class Spoofing:
    def __init__(self,model_path="C:/Users/yas/Desktop/PFE/models/spoof5000.pkl"):
        self.model = joblib.load(model_path)

    def preprocess_image(self, image):
        image = cv2.resize(image, (320, 320))
        img_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
        img_luv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        y_h = self.lbp_histogram(img_ycrcb[:,:,0]) # y channel
        cb_h = self.lbp_histogram(img_ycrcb[:,:,1]) # cb channel
        cr_h = self.lbp_histogram(img_ycrcb[:,:,2]) # cr channel
        
        h_h = self.lbp_histogram(img_luv[:,:,0]) # h channel
        s_h = self.lbp_histogram(img_luv[:,:,1]) # s channel
        v_h = self.lbp_histogram(img_luv[:,:,2]) # v channel

        feature_vector = np.concatenate((y_h,cb_h,cr_h,h_h,s_h,v_h))
        feature_vector = feature_vector.reshape(1, len(feature_vector))
        return feature_vector

    def calc_hist(self,img):
        histogram = [0] * 3
        for j in range(3):
            histr = cv2.calcHist([img], [j], None, [256], [0, 256])
            histr *= 255.0 / histr.max()
            histogram[j] = histr
        return np.array(histogram)

    def lbp_histogram(self,image,P=8,R=1,method = 'nri_uniform'):
        '''
        image: shape is N*M 
        '''
        lbp = feature.local_binary_pattern(image, P,R, method) # lbp.shape is equal image.shape
        # cv2.imwrite("lbp.png",lbp)
        max_bins = int(lbp.max() + 1) # max_bins is related P
        hist,_= np.histogram(lbp,  density=True, bins=max_bins, range=(0, max_bins))
        return hist
        # eps=1e-7
        # lbp = feature.local_binary_pattern(image, P,R, method="uniform")
        # (hist, _) = np.histogram(lbp.ravel(),
        # bins=np.arange(0, P + 3),range=(0, P + 2))
        # # normalize the histogram
        # hist = hist.astype("float")
        # hist /= (hist.sum() + eps)
        # # return the histogram of Local Binary Patterns
        # return hist
    
    def is_spoof(self, image):
        # Prétraiter l'image
        if image.size == 0:
            return None
        feature_vector = self.preprocess_image(image)

        # Faire une prédiction avec le modèle
        prediction = self.model.predict_proba(feature_vector)

        # Renvoyer la prédiction
        return prediction
