import cv2
from picamera2 import Picamera2
from threading import Thread, Event

class VideoStream:
    def __init__(self, stream_id=0):
        self.stream_id = stream_id
        self.vcap = Picamera2()
        self.vcap.preview_configuration.main.size = (696,360)
        self.vcap.preview_configuration.main.format = "RGB888"
        self.vcap.preview_configuration.controls.FrameRate=60
        self.vcap.preview_configuration.align()
        self.vcap.configure("preview")
        self.vcap.start()
        self.frames = []
        self.frame = self.vcap.capture_array()
        self.stopped = Event()
        self.frames.append(self.frame)
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True

    def start(self):
        self.stopped = Event()
        self.t.start()

    def update(self):
        while True:
            if self.stopped.is_set():
                break
            self.frame = self.vcap.capture_array()
            self.frames.append(self.frame)
            # Supprimer les frames non utilisées pour éviter l'accumulation
            if len(self.frames) > 30:
                self.frames.pop(0)

        self.vcap.release()

    def read(self):
        if len(self.frames) == 0 or self.stopped.is_set():
            return None
        
        frame = self.frames.pop(0)
        # Effectuer le traitement de l'image si nécessaire
        # frame_processed = ...
        
        return frame

    def stop(self):
        self.stopped.set()