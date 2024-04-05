from App import App
import cv2 ,time, os
import requests
import sys,getopt
<<<<<<< Updated upstream
from picamera2 import Picamera2
from dotenv import load_dotenv

load_dotenv()

def usage():
  print("Utilisation : {} -i <adresse_ip> -p <port> [-r <degre_rotation>]".format(sys.argv[0]))
  print("Options :")
  print("  -i <adresse_ip>  : Adresse IP du back")
  print("  -r <degre_rotation> : DegrÃƒÂ© de rotation de l'image (0, 90, 180 ou 270)")
  print("  -s <save faces> : Enregistrer de nouvelles images des visage (0 ou 1)")


=======
from centroidtracker2 import CentroidTracker
from VideoStream import VideoStream
import numpy as np
from scipy.signal import convolve2d as conv2
>>>>>>> Stashed changes

from skimage import color, data, restoration

def richardson_lucy_blind(image, psf, num_iter=50):    
    psf = np.ones((5, 5)) / 25
    rng = np.random.default_rng()
    # astro = conv2(image, psf, 'same')
    # Add Noise to Image
    

    # Restore Image using Richardson-Lucy algorithm
    deconvolved_RL = restoration.richardson_lucy(image, psf, num_iter=30)   
    return deconvolved_RL


rotations = {0:(None,(640,360 )),
             90:(cv2.ROTATE_90_CLOCKWISE,(360,640 )),
             180:(cv2.ROTATE_180,(640,360 )),
             270:(cv2.ROTATE_90_COUNTERCLOCKWISE,(360,640 ))
    }


if __name__ == '__main__':
<<<<<<< Updated upstream
    
    adresse_ip = os.getenv('URL_BACKEND')
    port = os.getenv('PORT')
    degre_rotation = int(os.getenv('ROTATION'))
=======

    adresse_ip = None
    port = None
    degre_rotation = 0
>>>>>>> Stashed changes
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

<<<<<<< Updated upstream
    # Create an instance of the PiCamera2 object
    cam = Picamera2()
    # Set the resolution of the camera preview
    cam.preview_configuration.main.size = degre_rotation[1]
    cam.preview_configuration.main.format = "RGB888"
    cam.preview_configuration.controls.FrameRate=60
    cam.preview_configuration.align()
    cam.configure("preview")
    cam.start()
    
=======
    # cap = VideoStream("C:/Users/yas/Downloads/archive (3)/20240228_192336 - abdeljalil abed.mp4")
    cap = VideoStream("C:/Users/yas/Desktop/test.avi")
    cap = VideoStream(0)
    cap.start()
>>>>>>> Stashed changes
    prev_frame_time = 0
    fp = []
    c = 0
    print("debut boucle")
    while True:
<<<<<<< Updated upstream
#         ret,frame=cap.read()
#         if not ret:
#             break
        #print("capture")
        frame = cam.capture_array()
        #resize frame to 0.5
        #frame = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_AREA)
        if not degre_rotation[0] == None:
            frame = cv2.rotate(frame, degre_rotation[0])
        
        c+=1
        # frame = cv2.resize(frame,(128,96))
        if c%5==0:
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
                    print(face.name, face.distance)    
                    r = requests.post(f"http://{adresse_ip}/postEtdsPresent",json={"matricule":face.name})
                    print(r)
                        #exit(0)
                    
                    
            #else:
                #   print("walo")
                    
            # else:
            #     print('unknown')
=======
        rects = []
        frame=cap.read()
        if frame is None:
            break
        
        # if not degre_rotation[0] == None:
        #     frame = cv2.rotate(frame, degre_rotation[0])
        
        c+=1
        # frame = cv2.resize(frame,None,fx=1.1,fy=1.1,interpolation=cv2.INTER_AREA)
        # frame = cv2.resize(frame,(320,320),interpolation=cv2.INTER_AREA)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # gamma = 1.5
        # lookup_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        # frame = cv2.LUT(frame, lookup_table)
        # frame, alpha, beta = automatic_brightness_and_contrast(frame)
        # recon = richardson_lucy_blind(frame,9)
        

        if c%1==0:
            
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

>>>>>>> Stashed changes
        
        new_frame_time = time.time()
        try: 
            fps = 1/(new_frame_time-prev_frame_time) 
        except (ZeroDivisionError):
            pass
        prev_frame_time = new_frame_time 
        fp.append(fps)
        fps = str(int(fps))
<<<<<<< Updated upstream
        #cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA) 
        
        
=======
        #cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)  
        cv2.imshow("lol",frame)
        # cv2.imshow("recon",recon)
        k = cv2.waitKey(1)
        if k == 27:         # wait for ESC key to exit
            break
            
>>>>>>> Stashed changes

    #average of fp
    c = 0
    for f in fp:
        c+=f
    try:
        print("average",c/len(fp))
    except:
        pass
    cv2.destroyAllWindows()
    



