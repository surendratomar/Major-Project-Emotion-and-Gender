from keras.models import load_model
from time import sleep
from tensorflow.keras.utils import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import os
import cvlib as cv





face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
classifier =load_model('./Emotion_Detection.h5')


# load model
model = load_model('gender_detection.model')

# open webcam
webcam = cv2.VideoCapture(0)
    
classes = ['man','woman']




class_labels = ['Angry','Happy','Neutral','Sad','Surprise','Fear','Disgust']

cap = cv2.VideoCapture(0)



while True:
    # Grab a single frame of video
    ret, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)


        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

        # make a prediction on the ROI, then lookup the class

            preds = classifier.predict(roi)[0]
            print("\nprediction = ",preds)
            label=class_labels[preds.argmax()]
            print("\nprediction max = ",preds.argmax())
            print("\nlabel = ",label)
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        else:
            cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        print("\n\n")
    cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        
        # loop through frames
        while webcam.isOpened():

            # read frame from webcam 
            status, frame = webcam.read()

            # apply face detection
            face, confidence = cv.detect_face(frame)


            # loop through detected faces
            for idx, f in enumerate(face):

                # get corner points of face rectangle        
                (startX, startY) = f[0], f[1]
                (endX, endY) = f[2], f[3]

                # draw rectangle over face
                cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)

                # crop the detected face region
                face_crop = np.copy(frame[startY:endY,startX:endX])

                if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
                    continue

                # preprocessing for gender detection model
                face_crop = cv2.resize(face_crop, (96,96))
                face_crop = face_crop.astype("float") / 255.0
                face_crop = img_to_array(face_crop)
                face_crop = np.expand_dims(face_crop, axis=0)

                # apply gender detection on face
                conf = model.predict(face_crop)[0] # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]

                # get label with max accuracy
                idx = np.argmax(conf)
                label = classes[idx]

                label = "{}: {:.2f}%".format(label, conf[idx] * 100)

                Y = startY - 10 if startY - 10 > 10 else startY + 10

                # write label and confidence above face rectangle
                cv2.putText(frame, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)

            # display output
            cv2.imshow("gender detection", frame)

            # press "Q" to stop
            if cv2.waitKey(1) & 0xFF == ord('e'):
                break


                
cap.release()
webcam.release()
cv2.destroyAllWindows()


























