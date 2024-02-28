from App import App
import cv2 ,time


if __name__ == '__main__':

    app = App("C:/Users/yas/Desktop/tempsdb")

    cap = cv2.VideoCapture(0)
    prev_frame_time = 0
    fp = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = app.extractFaces(frame)

    
        if len(faces)>0:
            for face in faces:
                frame = app.Draw(frame,face)
                # print(face.name,f"{dist:.3f}")
        # else:
        #     print('unknown')
        
        new_frame_time = time.time() 
        fps = 1/(new_frame_time-prev_frame_time) 
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