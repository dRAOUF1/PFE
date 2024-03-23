from typing import List, Tuple, Optional
import numpy as np
import cv2
import math

class Face:
    def __init__(
        self,
        x,
        y,
        x2,
        y2,
        confidence,
        left_eye=None,
        right_eye=None,
        nose=None,
        left_mouth=None,
        right_mouth=None
    ):
        """
        Initializes a Face object with the given parameters.

        Args:
            x (int): The x-coordinate of the top-left corner of the face bounding box.
            y (int): The y-coordinate of the top-left corner of the face bounding box.
            x2 (int): The x-coordinate of the bottom-right corner of the face bounding box.
            y2 (int): The y-coordinate of the bottom-right corner of the face bounding box.
            confidence (float): The confidence score of the face detection.
            left_eye (tuple, optional): The coordinates of the left eye. Defaults to None.
            right_eye (tuple, optional): The coordinates of the right eye. Defaults to None.
            nose (tuple, optional): The coordinates of the nose. Defaults to None.
            left_mouth (tuple, optional): The coordinates of the left mouth corner. Defaults to None.
            right_mouth (tuple, optional): The coordinates of the right mouth corner. Defaults to None.
        """
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

    def align_and_rotate(self,face_image):
        """
        Align and rotate the face coordinates based on the provided eye coordinates.

        Args:
            self.left_eye (tuple): Coordinates of the left eye after rotation.
            eye_right (tuple): Coordinates of the right eye after rotation.
        """
        # Calculate rotation angle
        dx = self.right_eye[0] - self.left_eye[0]
        dy = self.right_eye[1] - self.left_eye[1]
        angle = np.degrees(np.arccos(dx / np.linalg.norm((dx, dy)))) + 180

        # Adjust the angle based on the direction of the eyes
        if dy < 0:
            angle = -angle
        # Calculate center of rotation
        eye_center = (int((self.left_eye[0] + self.right_eye[0]) // 2), int((self.left_eye[1] + self.right_eye[1]) // 2))

        # Rotate face bounding box coordinates
        # self.x1, self.y1 = rotate_point((self.x1, self.y1), eye_center, angle)
        # self.x2, self.y2 = rotate_point((self.x2, self.y2), eye_center, angle)

        # Rotate facial keypoints coordinates
        if self.left_eye:
            self.left_eye = rotate_point(self.left_eye,eye_center, angle)
        if self.right_eye:
            self.right_eye = rotate_point(self.right_eye, eye_center, angle)
        if self.nose:
            self.nose = rotate_point(self.nose, eye_center, angle)
        if self.left_mouth:
            self.left_mouth = rotate_point(self.left_mouth, eye_center, angle)
        if self.right_mouth:
            self.right_mouth = rotate_point(self.right_mouth, eye_center, angle)

        rows, cols = face_image.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D(eye_center, angle, 1)
        rotated_face_image = cv2.warpAffine(face_image, rotation_matrix, (cols, rows))
        return rotated_face_image
    
    




    def toArray(self) -> np.ndarray:
        """
        Converts the Face object to a numpy array.

        Returns:
            numpy.ndarray: The array representation of the Face object.
        """
        try:
            return np.array([self.x1, self.y1, self.x2, self.y2, self.right_eye[0], self.right_eye[1], self.left_eye[0], self.left_eye[1], self.nose[0], self.nose[1], self.right_mouth[0], self.right_mouth[1], self.left_mouth[0], self.left_mouth[1], self.confidence])
        except (TypeError, IndexError):
            raise ValueError("Invalid face object. Some attributes are missing or have incorrect format.")


def rotate_point(point, origin, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    angle = math.radians(-angle)
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy
