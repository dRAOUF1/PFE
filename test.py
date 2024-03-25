from centroidtracker2 import CentroidTracker
from App import App
import cv2

# initialize our centroid tracker and frame dimensions
ct = CentroidTracker()
(H, W) = (None, None)
# load our serialized model from disk
print("[INFO] loading model...")
app = App("http://localhost:3001/getEmbeddings/34/1")
# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")
cap = cv2.VideoCapture("C:/Users/yas/Desktop/test.avi")
while True:
    rects = []
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    if W is None or H is None:
        (H, W) = frame.shape[:2]
    faces = app.extractFaces(frame)

    for face in faces:
        rects.append((int(face.x1), int(face.y1), int(face.x2), int(face.y2), "lol"))
        cv2.rectangle(frame, (int(face.x1), int(face.y1)), (int(face.x2), int(face.y2)), (0, 255, 0), 2)

    objects = ct.update(rects)

    for (objectID, centroid) in objects.items():
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(frame, str(text), (int(centroid[0]) - 10, int(centroid[1]) - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (int(centroid[0]), int(centroid[1])), 4, (0, 255, 0), -1)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(50) & 0xFF
	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()