from typing import List, Tuple, Optional
import numpy as np
class Face:
    def __init__(
        self,
        x,
        y,
        x2,
        y2,
        confidence,
        left_eye = None,
        right_eye = None,
        nose = None,
        left_mouth = None,
        right_mouth = None
    ):
        self.x1 = x
        self.y1 = y
        self.x2 = x2
        self.y2 = y2
        self.left_eye = left_eye
        self.right_eye = right_eye
        self.nose = nose
        self.left_mouth = left_mouth
        self.right_mouth = right_mouth
        self.confidence = confidence

    def toArray(self):
        return np.array([self.x1,self.y1,self.x2,self.y2,self.right_eye[0],self.right_eye[1],self.left_eye[0],self.left_eye[1],self.nose[0],self.nose[1],self.right_mouth[0],self.right_mouth[1],self.left_mouth[0],self.left_mouth[1],self.confidence])
    