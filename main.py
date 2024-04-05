from App import App
import cv2 ,time, os
import requests
import sys,getopt
from dotenv import load_dotenv
import gc

load_dotenv()

def usage():
  print("Utilisation : {} -i <adresse_ip> -p <port> [-r <degre_rotation>]".format(sys.argv[0]))
  print("Options :")
  print("  -i <adresse_ip>  : Adresse IP du back")
  print("  -r <degre_rotation> : DegrÃƒÂ© de rotation de l'image (0, 90, 180 ou 270)")
  print("  -s <save faces> : Enregistrer de nouvelles images des visage (0 ou 1)")



from centroidtracker2 import CentroidTracker
from VideoStream import VideoStream
import numpy as np



rotations = {0:(None,(1920,1080)),
             90:(cv2.ROTATE_90_CLOCKWISE,(1080,1920 )),
             180:(cv2.ROTATE_180,(1920,1080)),
             270:(cv2.ROTATE_90_COUNTERCLOCKWISE,(1080,1920 ))
    }


if __name__ == '__main__':
    gc.enable()
    
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

    print("fin app build")
    presents = []

    cap = VideoStream(0)
    cap.start()
    ct = CentroidTracker()
    
    prev_frame_time = 0
    fp = []
    c = 0
    print("debut boucle")
    while True:
        rects = []
        frame=cap.read()
        if frame is None:
            continue
        
        if not degre_rotation[0] == None:
            frame = cv2.rotate(frame, degre_rotation[0])
        
        c+=1
        if c%2==0:
            
            faces = app.find_match(frame)
            for face in faces:
                if save_faces:
                    cropped = frame[int(face.face.y1):int(face.face.y2), int(face.face.x1):int(face.face.x2)]
                    cropped = cv2.resize(cropped,(128,128))
                    cv2.imwrite(f"faces/{face.name}-{time.time()}.jpg",cropped)
                rects.append((int(face.face.x1), int(face.face.y1), int(face.face.x2), int(face.face.y2), face.name))
                frame = app.Draw(frame,face)
                # print("/n",face.name, face.distance )
                # print(face.name)    
            objects = ct.update(rects)
                # r = requests.post(f"http://{adresse_ip}:{port}/postEtdsPresent",json={"matricule":face.name})
                # print(r)
                    #exit(0)
        
        new_frame_time = time.time()
        try: 
            fps = 1/(new_frame_time-prev_frame_time) 
        except (ZeroDivisionError):
            pass
        prev_frame_time = new_frame_time 
        fp.append(fps)
        fps = str(int(fps))
        # cv2.imshow("lol",frame)
        gc.collect()
        # cv2.imshow("recon",recon)
        k = cv2.waitKey(1)
        if k == 27:         # wait for ESC key to exit
            break
            
    #average of fp
    c = 0
    for f in fp:
        c+=f
    try:
        print("average",c/len(fp))
    except:
        pass
    cv2.destroyAllWindows()
    



