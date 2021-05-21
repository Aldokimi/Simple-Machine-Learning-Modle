
import os
import cv2 as cv
import numpy as np

#______________________________________________Instantiating_____________________________________________#

#note: The names in the list should be exactly like the file names which contain the images
characters = ['Gus Fring', 'Hank Schrader', 'Jesse Pinkman', 'Saul Goodman', 'Skyler White' , 'Walter White' , 'Walter White Jr' ]

DIR = r'./Characters' # The dierctory of the characters images files

# A pre-trained openCV face recognizer ( classifier ) that recognize if there is a face in the image or not  
haar_cascade = cv.CascadeClassifier('haar_face.xml')    

features = []   # The set of our faces 
labels = []     # The corosponding laple for every face in the features list

#______________________________________________Training___________________________________________________#

# This fuction will loop over every folder in the folder specified by DIR and inside every folder it is going to 
# loop over every image and grap the face in that image and add that face to a training set  
def create_train():
    for person in characters:
        path = os.path.join(DIR, person)        #Getting the path of the image folder
        label = characters.index(person)        

        for img in os.listdir(path):            #Looping over the images 
            img_path = os.path.join(path,img)   #Getting the path of the image

            img_array = cv.imread(img_path)     #Reading the image
            if img_array is None:               
                continue 
                
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)    #Converting the image to the gray scale 

            #Recognizing the faces in the images by using haar_cascade classifier
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            
            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]  #Grap the faces ragion of interest and crop it 
                features.append(faces_roi)      
                labels.append(label)

create_train()
print('Training done ---------------')

#______________________________________Instantiating the recognizer_____________________________________#

#Creating a numby arrays for our trained images 
features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()   #Instantiating our face recognizer

# Train the Recognizer on the features list and the labels list
face_recognizer.train(features,labels)

face_recognizer.save('face_trained.yml') # A file which contain all the recognized faced so we can use them later
np.save('features.npy', features)
np.save('labels.npy', labels)