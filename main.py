from App import App
import cv2 ,time
import requests 

if __name__ == '__main__':
    displayed_ids=[]
    app = App("C:/Users/TRETEC/Desktop/PFE/archive")

    cap = cv2.VideoCapture(0)
    prev_frame_time = 0
    fp = []
    c = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        c+=1
        # frame = cv2.rescale(frame,(128,96))
        if c%1==0:
            faces = app.find_match(frame)

        
            if len(faces)>0:
                for face in faces:
                    frame = app.Draw(frame,face)
                    # print(face.name)
                    if face.name not in displayed_ids:
                        displayed_ids.append(face.name)
                        r = requests.post('http://localhost:3001/postEtds',json={'matricule':face.name})
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