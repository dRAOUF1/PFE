import cv2
import numpy as np
from VideoStream import VideoStream
from App import App

# cap = VideoStream("C:/Users/yas/Desktop/test.avi")
#cap.start()
app = App("http://localhost:3001/getEmbeddings/all/all")

im1 = cv2.imread("C:/Users/yas/Desktop/tempsdb/ayoub.jpg")
face = app.extractFaces(im1)
im1 = im1[int(face[0].y1):int(face[0].y2),int(face[0].x1):int(face[0].x2)]
im1 = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
im2 = cv2.imread("C:/Users/yas/Desktop/tempsdb/sofiane.jpg")
face = app.extractFaces(im2)
print(len(face))
im2 = im2[int(face[0].y1)-10:int(face[0].y2)+10,int(face[0].x1)-10:int(face[0].x2)+10]
im2 = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
cv2.imshow("im1",im1)
cv2.imshow("im2",im2)
cv2.waitKey(0)
#extract sift intrest points
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(im1,None)
kp2, des2 = sift.detectAndCompute(im2,None)

bf = cv2.BFMatcher(
    # cv2.NORM_L2, 
    # crossCheck=True
    )

# Match descriptors.
# matches = bf.match(des1,des2)
matches=bf.knnMatch(des1,des2,k=2)
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

print(len(matches))
print(len(good))