from typing import List, Tuple, Optional
import numpy as np
class Face:
    x: int
    y: int
    w: int
    h: int
    left_eye: Tuple[int, int]
    right_eye: Tuple[int, int]
    nose: Tuple[int, int]
    left_mouth: Tuple[int, int]
    right_mouth: Tuple[int, int]
    confidence: float

    def __init__(
        self,
        x: int,
        y: int,
        x2: int,
        y2: int,
        left_eye: Optional[Tuple[int, int]] = None,
        right_eye: Optional[Tuple[int, int]] = None,
        nose: Optional[Tuple[int, int]] = None,
        left_mouth: Optional[Tuple[int, int]] = None,
        right_mouth: Optional[Tuple[int, int]] = None,
        confidence: Optional[float] = None,
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
    