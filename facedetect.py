#1) Read a video from a web cam using opencv
#2) Face detection in a video
#3) click 20 picture of the person who comes in the front of camera and save them as numpy

import cv2
import numpy as np

#create a camera object
cam = cv2.VideoCapture(0)

#Ask the name
person_name = input("Enetr the name of person : ")
dataset_path = "./data/"
offset=20
faceData = []
skip = 0
#model
model = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

#read image from camera object
while True:
    success, img = cam.read()
    if not success:
        print("raeding camera error!")

    #store the gray image
    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = model.detectMultiScale(img,1.3,5)

    #sorting the facewith largest bounding box
    faces = sorted(faces, key= lambda f:f[2]*f[3])
    
    # pick the largets faces
    if len(faces)>0 : 
        f = faces[-1]
    
        x,y,w,h, = f
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,255), 2)

        #crop and save the largest face
        cropped_face = img[y-offset : y+h+offset, x-offset : x+w+offset]
        cropped_face = cv2.resize(cropped_face, (100,100))
        skip += 1
        if skip % 10 == 0:
            faceData.append(cropped_face)
            print("saved so far" + str(len(faceData)))

    cv2.imshow("Image Window", img)
    #cv2.imshow("cropped face", cropped_face)
    
    key = cv2.waitKey(1) #pause here for 1 ms before you read the next image
    if key == ord('q'):
        break

#write the faceData on the Disk
faceData = np.asarray(faceData)
m = faceData.shape[0]
faceData = faceData.reshape((m,-1))

print(faceData.shape)

#save as the Disk as np array
filepath = dataset_path + person_name + ".npy"
np.save(filepath, faceData)
print("DATA SAVED SUCCESSFULLY : " + filepath)

#release Camera and destroy Window
cam.release()
cv2.destroyAllWindows()