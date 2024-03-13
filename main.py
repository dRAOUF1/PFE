from App import App
import cv2 ,time
import requests
import sys,getopt


def usage():
  print("Utilisation : {} -i <adresse_ip> -p <port> [-r <degre_rotation>]".format(sys.argv[0]))
  print("Options :")
  print("  -i <adresse_ip>  : Adresse IP du back")
  print("  -p <port>  : port")
  print("  -r <degre_rotation> : DegrÃ© de rotation de l'image (0, 90, 180 ou 270)")
  print("  -s <save faces> : Enregistrer de nouvelles images des visage (0 ou 1)")





rotations = {0:(None,(640,360 )),
             90:(cv2.ROTATE_90_CLOCKWISE,(360,640 )),
             180:(cv2.ROTATE_180,(640,360 )),
             270:(cv2.ROTATE_90_COUNTERCLOCKWISE,(360,640 ))
    }

if __name__ == '__main__':
    
    adresse_ip = None
    port = None
    degre_rotation = 0
    save_faces = False

    # Analyser les arguments de la ligne de commande
    try:
        opts, args = getopt.getopt(sys.argv[1:], "i:r:p:")
    except getopt.GetoptError as err:
        print(err)
        usage()
        sys.exit(1)
    #print('hado opts',opts)
    # Parcourir les options et les arguments
    for opt, arg in opts:
        if opt in ("-i"):
          adresse_ip = arg
        elif opt in ("-r", "--rotation"):
          degre_rotation = int(arg)
        elif opt in ("-p", "--port"):
            port = arg
        elif opt in ("-s"):
            if opt == 1:
                save_faces = True
        
        else:
          print("Option inconnue : {}".format(opt))
          usage()
          sys.exit(1)

    # VÃ©rifier le degrÃ© de rotation
    if degre_rotation not in (0, 90, 180, 270):
        print("Degré de rotation invalide : {}".format(degre_rotation))
        usage()
        sys.exit(1)
    else:
        degre_rotation = rotations[degre_rotation]
    if adresse_ip == None:
        print("Adresse ip invalide")
        sys.exit(1)
    if port == None:
        print("port invalide")
        sys.exit(1)
    
    app = App("http://localhost:3001/getEmbeddings/34/1")
    print("fin app build")
    presents = []

    cap = cv2.VideoCapture(0)
    prev_frame_time = 0
    fp = []
    c = 0
    print("debut boucle")
    # while True:
        # ret,frame=cap.read()
        # if not ret:
        #     break
        #print("capture")
    frame = cv2.imread("C:/Users/yas/Desktop/abadlijpg.jpg")
    #resize frame to 0.5
    frame = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_AREA)
    if not degre_rotation[0] == None:
        frame = cv2.rotate(frame, degre_rotation[0])
    
    c+=1
    # frame = cv2.resize(frame,(128,96))
    if c%1==0:
        faces = app.find_match(frame)

        #print(len(faces),"hadi len of faces")
        if len(faces)>0:
            for face in faces:
                if save_faces:
                    cropped = frame[int(face.face.y1):int(face.face.y2), int(face.face.x1):int(face.face.x2)]
                    cropped = cv2.resize(cropped,(128,128))
                    cv2.imwrite(f"faces/{face.name}-{time.time()}.jpg",cropped)
                frame = app.Draw(frame,face)
                # print("\n",face.name,matricule[face.name] )
                print(face.name)    
                if face.name not in presents:
                    presents.append(face.name)
                    r = requests.post(f"http://{adresse_ip}:{port}/postEtdsPresent",json={"matricule":face.name})
                    print(r)
                    #exit(0)
                
                
        #else:
            #   print("walo")
                
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
    #cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)  
    cv2.imshow("lol",frame)
    k = cv2.waitKey(0)
        # if k == 27:         # wait for ESC key to exit
        #     break
        

    #average of fp
    c = 0
    for f in fp:
        c+=f
    try:
        print("average",c/len(fp))
    except:
        pass
    #cv2.destroyAllWindows()
    

