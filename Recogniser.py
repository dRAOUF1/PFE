import numpy as np
import cv2


class Recogniser:
    def __init__(self, modelPath, disType=0):
        self._modelPath = modelPath
        self._model = cv2.FaceRecognizerSF.create(
            model=self._modelPath,
            config="",
            backend_id=0,
            target_id=0)

        self._disType = disType # 0: cosine similarity, 1: Norm-L2 distance
        assert self._disType in [0, 1], "0: Cosine similarity, 1: norm-L2 distance, others: invalid"

        self._threshold_cosine = 0.363
        self._threshold_norml2 = 1.128


    def _preprocess(self, image, bbox):
        if bbox is None:
            return image
        else:
            return self._model.alignCrop(image, bbox)

    def infer(self, image, bbox=None):
        # Preprocess
        inputBlob = self._preprocess(image, bbox)
        # Forward
        features = self._model.feature(inputBlob)
        return features

    def match(self, image1, face1, image2, face2):
        feature1 = self.infer(image1, face1)
        feature2 = self.infer(image2, face2)

        if self._disType == 0: # COSINE
            cosine_score = self._model.match(feature1, feature2, self._disType)
            return cosine_score if cosine_score >= self._threshold_cosine else 0
        else: # NORM_L2
            norml2_distance = self._model.match(feature1, feature2, self._disType)
            return norml2_distance if norml2_distance <= self._threshold_norml2 else 0
        
    def match(self, feature1, feature2):
        if self._disType == 0: # COSINE
            cosine_score = self._model.match(feature1, feature2, self._disType)
            return cosine_score if cosine_score >= self._threshold_cosine else 0
        else: # NORM_L2
            norml2_distance = self._model.match(feature1, feature2, self._disType)
            return norml2_distance if norml2_distance <= self._threshold_norml2 else 0
