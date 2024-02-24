from typing import List, Tuple, Optional
class Face:
    x: int
    y: int
    w: int
    h: int
    left_eye: Tuple[int, int]
    right_eye: Tuple[int, int]
    confidence: float

    def __init__(
        self,
        x: int,
        y: int,
        x2: int,
        y2: int,
        left_eye: Optional[Tuple[int, int]] = None,
        right_eye: Optional[Tuple[int, int]] = None,
        confidence: Optional[float] = None,
    ):
        self.x = x
        self.y = y
        self.x2 = x2
        self.y2 = y2
        self.left_eye = left_eye
        self.right_eye = right_eye
        self.confidence = confidence