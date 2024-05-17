#1) Read a video from a web cam using opencv
#2) Face detection in a video
import cv2

#create a camera object
cam = cv2.VideoCapture(0)

#model
model = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

#read image from camera object
while True:
    success, img = cam.read()
    if not success:
        print("raeding camera error!")

    faces = model.detectMultiScale(img,1.3,5)

    for f in faces:
        x,y,w,h, = f
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,255), 1)
    
    cv2.imshow("Image Window", img)
    key = cv2.waitKey(1) #pause here for 1 ms before you read the next image
    if key == ord('q'):
        break

#release Camera and destroy Window
cam.release()
cv2.destroyAllWindows()