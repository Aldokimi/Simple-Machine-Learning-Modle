
import numpy as np
import cv2 as cv
import math  

haar_cascade = cv.CascadeClassifier('haar_face.xml')

characters = ['Gus Fring', 'Hank Schrader', 'Jesse Pinkman', 'Saul Goodman', 'Skyler White' , 'Walter White' , 'Walter White Jr' ]

features = np.load('features.npy', allow_pickle=True)
# labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')


print('Enter the image path: \nNote: the image should be for one of those people: \nGus Fring\nHank Schrader\nJesse Pinkman\nSaul Goodman\nSkyler White\nWalter White\nWalter White Jr\nPath: ')
test_img_path = str(input())
img = cv.imread(test_img_path)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#cv.imshow('Person', gray)

# Detecting the face in the image by using our classifier
faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h,x:x+w]

    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {len(features)} with a confidence of {confidence}')

    cv.putText(img,str(characters[label] + ' ' + str(math.floor(confidence)) + '%'), (x-5,y-5), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=1)
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow('Detected Face', img)

cv.waitKey(0)