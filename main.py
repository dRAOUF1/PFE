from App import App
import cv2 ,time


if __name__ == '__main__':

    app = App("C:/Users/yas/Desktop/tempyolov8")

    cap = cv2.VideoCapture("C:/Users/yas/Downloads/djalilcap.mov")
    prev_frame_time = 0
    fp = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        dist,faces = app.find_match(frame)

    
        if len(faces)>0:
            for face in faces:
                frame = app.Draw(frame,face)
                print(face.name,f"{dist:.3f}")
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
    # Instantiate SFace for face recognition
    # # Instantiate YuNet for face detection

    # img1 = cv2.imread("./zouj.jpg")
    # # img2 = cv2.imread(args.input2)

    # # Detect faces
    # detector.setInputSize([img1.shape[1], img1.shape[0]])
    # face1 = detector.infer(img1)
    # recognizer.infer(img1,face1[0].toArray()[:-1])













    # assert len(face1) > 0, 'Cannot find a face in {}'.format("C:/Users/yas/Desktop/tempyolov8/amina.jpg")
    # cap = cv2.VideoCapture("C:/Users/yas/Downloads/djalil.mov")
    # while True:
    #     ret, img2 = cap.read()
    #     if not ret:
    #         break
    #     detector.setInputSize([img2.shape[1], img2.shape[0]])
    #     face2 = detector.infer(img2)
    #     if len(face2)> 0:
    #         img2 = detector.Draw(img2, face2)
    #     cv2.imshow("camera input", img2)
    #     # if not face2.shape[0] > 0:
    #     #     print('Cannot find a face in {}'.format("camera input"))

    #     # Match
    #     # try:
    #     #     result = recognizer.match(img1, face1[0][:-1], img2, face2[0][:-1])
    #     #     print('Result: {}.'.format('same identity' if result else 'different identities'))
    #     #     if result:
    #     #         print('Confidence: {:.2f}.'.format(recognizer.confidence))
    #     #         break
    #     # except (IndexError):
    #     #     print('Cannot find a face in {}'.format("camera input"))

        
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
