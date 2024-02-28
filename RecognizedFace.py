from Face import Face

class RecognizedFace:
    def __init__(self, name, distance, face: Face):
        self.name = name
        self.distance = distance
        self.face = face

    
