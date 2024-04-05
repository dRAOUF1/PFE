from App import App
import cv2 ,time, os
import requests
import sys,getopt
from picamera2 import Picamera2
from dotenv import load_dotenv
import random
from centroidtracker2 import CentroidTracker

load_dotenv()

def usage():
  print("Utilisation : {} -i <adresse_ip> -p <port> [-r <degre_rotation>]".format(sys.argv[0]))
  print("Options :")
  print("  -i <adresse_ip>  : Adresse IP du back")
  print("  -r <degre_rotation> : DegrÃƒÂ© de rotation de l'image (0, 90, 180 ou 270)")
  print("  -s <save faces> : Enregistrer de nouvelles images des visage (0 ou 1)")





rotations = {0:(None,(640,360 )),
             90:(cv2.ROTATE_90_CLOCKWISE,(360,640 )),
             180:(cv2.ROTATE_180,(640,360 )),
             270:(cv2.ROTATE_90_COUNTERCLOCKWISE,(360,640 ))
    }

if __name__ == '__main__':
    
    adresse_ip = os.getenv('URL_BACKEND')
    port = os.getenv('PORT')
    degre_rotation = int(os.getenv('ROTATION'))
    save_faces = False

    # Analyser les arguments de la ligne de commande
    try:
        opts, args = getopt.getopt(sys.argv[1:], "ir")
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
        elif opt in ("-s"):
            if opt == 1:
                save_faces = True
        
        else:
          print("Option inconnue : {}".format(opt))
          usage()
          sys.exit(1)

    # VÃƒÂ©rifier le degrÃƒÂ© de rotation
    if degre_rotation not in (0, 90, 180, 270):
        print("DegrÃ© de rotation invalide : {}".format(degre_rotation))
        usage()
        sys.exit(1)
    else:
        degre_rotation = rotations[degre_rotation]
    
    
    app = App(f"http://{adresse_ip}/getEmbeddings/all/all")
    print(app.embeddings)
    print("fin app build")
    presents = []

    # Create an instance of the PiCamera2 object
    cam = Picamera2()
    # Set the resolution of the camera preview
    cam.preview_configuration.main.size = (1080,720)
    cam.preview_configuration.main.format = "RGB888"
    cam.preview_configuration.controls.FrameRate=30
    cam.preview_configuration.align()
    cam.configure("preview")
    cam.start()
    
    prev_frame_time = 0
    fp = []
    c = 0
    ran = random.randint(1,1000)
    ct = CentroidTracker()

    with open(f"log-{ran}.txt","w") as f:
        print("debut boucle")
        while True:
    #         ret,frame=cap.read()
    #         if not ret:
    #             break
            #print("capture")
            rects = []
            frame = cam.capture_array()
            
            #frame = cv2.resize(frame,(350,140))
            
            #resize frame to 0.5
            #frame = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_AREA)
            if not degre_rotation[0] == 0:
                rotated_frame = cv2.rotate(frame, degre_rotation[0])
                
            
            #time.sleep(1)
            c+=1
            # frame = cv2.resize(frame,(128,96))
            if c%2==0:
                rects = []
                faces = app.find_match(rotated_frame)

                #print(len(faces),"hadi len of faces")
                if len(faces)>0:
                    for face in faces:
                        if save_faces:
                            cropped = frame[int(face.face.y1):int(face.face.y2), int(face.face.x1):int(face.face.x2)]
                            cropped = cv2.resize(cropped,(128,128))
                            cv2.imwrite(f"faces/{face.name}-{time.time()}.jpg",cropped)
                        rects.append((int(face.face.x1), int(face.face.y1), int(face.face.x2), int(face.face.y2), face.name))
                        # frame = app.Draw(frame,face)
                        # print("\n",face.name,matricule[face.name] )
                        print(face.name, face.distance)
                        
                        #f.write(f"{face.name}    {face.distance}\n")
                        
                        # r = requests.post(f"http://{adresse_ip}/postEtdsPresent",json={"matricule":face.name})
                        # print(r)
                            #exit(0)
                objects = ct.update(rects)
                # print(objects)
            cv2.imshow('frame',rotated_frame)
            if cv2.waitKey(1)==27:
                break
                 
            
            new_frame_time = time.time()
            try: 
                fps = 1/(new_frame_time-prev_frame_time) 
            except (ZeroDivisionError):
                pass
            prev_frame_time = new_frame_time 
            fp.append(fps)
            fps = str(int(fps))
        #cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA) 
        
        

    #average of fp
    c = 0
    for f in fp:
        c+=f
    try:
        print("average",c/len(fp))
    except:
        pass
    cv2.destroyAllWindows()
    



