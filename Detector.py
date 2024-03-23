import numpy as np
import cv2, dlib
from Face import Face

class Detector:
    """
    A class that represents a face detector.

    Attributes:
        _modelPath (str): The path to the face detection model.
        _inputSize (tuple): The input size of the model in the format (width, height).
        _confThreshold (float): The confidence threshold for face detection.
        _nmsThreshold (float): The non-maximum suppression threshold for face detection.
        _model (cv2.FaceDetectorYN): The face detection model.


    Methods:
        __init__(self, modelPath, inputSize=[320, 320], confThreshold=0.6, nmsThreshold=0.3):
            Initializes the Detector object with the specified parameters.
        setInputSize(self, input_size):
            Sets the input size of the model.
        infer(self, image):
            Performs face detection on the given image and returns a list of detected faces.
    """

    def __init__(self, modelPath, predictor_path,inputSize=[320, 320], confThreshold=0.6, nmsThreshold=0.3):
        """
        Initializes the Detector object with the specified parameters.

        Args:
            modelPath (str): The path to the face detection model.
            inputSize (list, optional): The input size of the model in the format [width, height]. Defaults to [320, 320].
            confThreshold (float, optional): The confidence threshold for face detection. Defaults to 0.6.
            nmsThreshold (float, optional): The non-maximum suppression threshold for face detection. Defaults to 0.3.
        """
        self._modelPath = modelPath
        self._inputSize = tuple(inputSize) # [w, h]
        self._confThreshold = confThreshold
        self._nmsThreshold = nmsThreshold
        self._predictor = dlib.shape_predictor("./models/shape_predictor_5_face_landmarks.dat")
        self._model = cv2.FaceDetectorYN.create(
            model=self._modelPath,
            config="",
            input_size=self._inputSize,
            score_threshold=self._confThreshold,
            nms_threshold=self._nmsThreshold,
            top_k=5000,
            backend_id=0,
            target_id=0)


    def setInputSize(self, input_size):
        """
        Sets the input size of the model.

        Args:
            input_size (list): The input size of the model in the format [width, height].
        """
        self._model.setInputSize(tuple(input_size))

    def infer(self, image):
        """
        Performs face detection on the given image and returns a list of detected faces.

        Args:
            image: The image on which to perform face detection.

        Returns:
            list: A list of detected faces, where each face is represented as a Face object.
        """
        # Forward
        results = self._model.detect(image)
        if results[1] is not None:
            
            faces = [ Face(
                x=face[0],
                y=face[1],
                x2=face[0] + face[2],
                y2=face[1] + face[3],
                confidence=face[14],
                right_eye=(face[4], face[5]),
                left_eye=(face[6], face[7]),
                nose=(face[8], face[9]),
                right_mouth=(face[10], face[11]), # (x, y)
                left_mouth=(face[12], face[13]) # (x, y
            ) for face in results[1] if face[2]>27 and face[3]>27 and face[0]>0 and face[1]>0 and face[0]+face[2]>0 and face[1]+face[3]>0]

            # for face in faces:
            #     rect = dlib.rectangle(face.x1,face.y1,face.x2,face.y2)
            #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            #     shape = self._predictor(gray, rect)
            #     shape = self._shape_to_np(shape)
                
            #     #find the center between two points
                
            #     face.left_eye = ((shape[0][0]+shape[1][0])//2, (shape[0][1]+shape[1][1])//2)
            #     face.right_eye = ((shape[2][0]+shape[3][0])//2, (shape[2][1]+shape[3][1])//2)
            return faces
        
        return []

    def _shape_to_np(self,dlib_shape, dtype="int"):
        """Converts dlib shape object to numpy array"""

        # Initialize the list of (x,y) coordinates
        coordinates = np.zeros((dlib_shape.num_parts, 2), dtype=dtype)

        # Loop over all facial landmarks and convert them to a tuple with (x,y) coordinates:
        for i in range(0, dlib_shape.num_parts):
            coordinates[i] = (dlib_shape.part(i).x, dlib_shape.part(i).y)

        # Return the list of (x,y) coordinates:
        return coordinates