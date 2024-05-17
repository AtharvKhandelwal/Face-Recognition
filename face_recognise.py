import cv2
import numpy as np
import os

# Data Prep
dataset_path = "./data/"
faceData = []
labels = []
nameMap = {}

classId = 0

for f in os.listdir(dataset_path):
    if f.endwith(".npy"):

        nameMap[classId] = f[ :-4]
        #X-values
        dataitem = np.load(dataset_path + f)
        m = dataitem.shape[0]
        faceData.append(dataitem)
        # print(dataitem.shape)
        
        #Y-value
        target = classId * np.ones((m,))
        classId += 1
        labels.append(target)

# print(faceData)
# print(labels)

xt = np.concatenate(faceData, axis=0)
yt = np.concatenate(labels, axis=0).reshape((-1,1))
print(xt.shape)
print(yt.shape)
print(nameMap)

# Algorithm
def dist(p,q):
    return np.sqrt(np.sum((p-q)**2))

def KNN(x,y,xt,k=5):

    m = x.shape[0]
    dlist = []

    for i in range(m):
        d = dist(x[i], xt)
        dlist.append((d,y[i]))

    dlist = sorted(dlist)
    dlist = np.array(dlist[:k])
    labels = dlist[:,1]

    labels, cnts = np.unique(labels, return_counts=True)
    idx = cnts.argmax()
    pred = labels[idx]

    return int(pred)

# Predictions (PART - 3)

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

    # render a box around each face nd predict its face
    for f in faces:
        x,y,w,h, = f
        print(f)
        # cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,255), 2)
        #crop the faces
        cropped_face = img[y-20 : y+h+20, x-20 : x+w+20]
        cropped_face = cv2.resize(cropped_face, (100,100))
        
        #predict the name using--------KNN----------
        classpredicted = KNN(xt,yt,cropped_face.flatten())
        # NMAE  
        namepredicted = nameMap[classpredicted]
        print(namepredicted)
        #Dispaly the name and Box
        cv2.putText(img, namepredicted, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,255), 2)

    cv2.imshow("Prediction Window", img)

    key = cv2.waitKey(1) #pause here for 1 ms before you read the next image
    if key == ord('q'):
        break

#release Camera and destroy Window
cam.release()
cv2.destroyAllWindows()