from App import App
import cv2 ,time


if __name__ == '__main__':

    app = App("C:/Users/yas/Desktop/tempsdb")

    cap = cv2.VideoCapture("C:/Users/yas/Downloads/ayoub.mov")
    prev_frame_time = 0
    fp = []
    c = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        c+=1
        # frame = cv2.rescale(frame,(128,96))
        if c%2==0:
            faces = app.extractFaces(frame)

        
            if len(faces)>0:
                for face in faces:
                    frame = app.Draw(frame,face)
                    # print(face.name)
            # else:
            #     print('unknown')
        
        new_frame_time = time.time()
        try: 
            fps = 1/(new_frame_time-prev_frame_time) 
        except (ZeroDivisionError):
            pass
        prev_frame_time = new_frame_time 
        fp.append(fps)
        fps = str(int(fps))
        cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)  
        cv2.imshow("lol",frame)
        k = cv2.waitKey(10)
        if k == 27:         # wait for ESC key to exit
            break
        

    #average of fp
    c = 0
    for f in fp:
        c+=f
    print("average",c/len(fp))
    cap.release()