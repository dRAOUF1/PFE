import cv2
from picamera2 import Picamera2
from threading import Thread, Event

# defining a helper class for implementing multi-threading 
class VideoStream:
    # initialization method 
    def __init__(self, stream_id=0):
        self.stream_id = stream_id # default is 0 for main camera 
        
        # opening video capture stream 
        self.vcap = Picamera2()
        # Set the resolution of the camera preview
        self.vcap.preview_configuration.main.size = (1920,1080)
        self.vcap.preview_configuration.main.format = "RGB888"
        self.vcap.preview_configuration.controls.FrameRate=60
        self.vcap.preview_configuration.align()
        self.vcap.configure("preview")
        self.vcap.start()

        # if self.vcap.isOpened() is False :
        #     print("[Exiting]: Error accessing webcam stream.")
        #     exit(0)
        # fps_input_stream = int(self.vcap.get(5)) # hardware fps
        # print("FPS of input stream: {}".format(fps_input_stream))
            
        self.frames = []
        # reading a single frame from vcap stream for initializing 
        self.frame = self.vcap.capture_array()
        
        # self.stopped is initialized to False 
        # self.stopped = Event()  
        self.frames.append(self.frame)
        # thread instantiation  
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True # daemon threads run in background 
        
    # method to start thread 
    def start(self):
        self.stopped = Event()
        self.t.start()
    # method passed to thread to read next available frame  
    def update(self):
        while True :
            if self.stopped.is_set() :
                break
            self.frame = self.vcap.capture_array()
            
            self.frames.append(self.frame)
        self.vcap.release()
    # method to return latest read frame 
    def read(self):
        if len(self.frames) == 0 and self.stopped.is_set():
            return None
        
        return self.frames.pop(0)
    # method to stop reading frames 
    def stop(self):
        self.stopped.set()