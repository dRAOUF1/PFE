from ultralytics import YOLO
import cv2
import numpy as np
import time

fps=0
pos=(30,60) #top-left
font=cv2.FONT_HERSHEY_COMPLEX
height=1.5 #font_scale
color=(0,0,255) #text color, OpenCV operates in BGR- RED
weight=3   #font-thickness

# model1 = YOLO("models\yolov8n_100e.pt")
cap = cv2.VideoCapture(0)
# frame = cv2.imread("ana.jpg")
# #preprocess the image
# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# frame = cv2.resize(frame, (640, 480))
# frame = frame/255.0

model2 = YOLO("models\yolov8n-face.pt")

# model3 = YOLO("models/best.pt")

#model4 = YOLO("models/best.onnx",task="detect")
#model5 = YOLO("models/yolov8n-face.onnx",task="detect")
#model6 = YOLO("models\yolov8-lite-s.pt",task="detect")

c=0
while True:
    try:
        tStart=time.time()
        ret, frame = cap.read()
        cv2.putText(frame,str(round(fps))+' FPS',pos,font,height,color,weight)
        if (c%10==0):
            results = model2.predict(frame,stream=True)
            boxes = results[0].boxes
            x1, y1, x2, y2= int(boxes.xyxy[0][0]),int(boxes.xyxy[0][1]),int(boxes.xyxy[0][2]),int(boxes.xyxy[0][3])
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    except:
        pass
    cv2.imshow('frame', frame)
    k = cv2.waitKey(10)
    if k == 27:         # wait for ESC key to exit
        break
    tEnd=time.time()
    looptime=tEnd-tStart
    fps=1/looptime
    c+=1