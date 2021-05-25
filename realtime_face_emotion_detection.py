# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 13:37:23 2020

@author: Shreya Jaitley
"""
#importing the required libraries
import cv2
import face_recognition
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json

 
cap = cv2.VideoCapture(1)
# Check if the webcam is opened correctly
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

#load the model and load the weights
face_exp_model=model_from_json(open("code/dataset/facial_expression_model_structure.json","r").read())  
#load weights into model  
face_exp_model.load_weights('code/dataset/facial_expression_model_weights.h5')    
#list of emotion models
emotions_label= ('angry','disgust','fear','happy','sad','surprise','neutral')
    
#initailize the variables to hold all face loacations in the frame
all_face_locations=[]

#create a while loop
while True:
    #get current frame from video stream as an image
    ret, current_frame=cap.read()
    #resizing the current frame to 1/4th the orignal frame to process faster
    current_frame_small = cv2.resize(current_frame, (0,0), fx=0.25, fy=0.25)
    #detect all faces in image
    all_face_locations=face_recognition.face_locations(current_frame_small, number_of_times_to_upsample=2, model="hog")
    
    #looping through the face locations
    
    for index,current_face_location in enumerate(all_face_locations):
        #splitting the tuple to get four position values
        top_pos,right_pos,bottom_pos,left_pos=current_face_location
        
        #change the position magnitude to fit the actual size vieo frame
        top_pos=top_pos*4
        right_pos=right_pos*4
        bottom_pos=bottom_pos*4
        left_pos=left_pos*4
        print('Found face {} at top:{},right:{},bottom:{},left:{}'.format(index+1,top_pos,right_pos,bottom_pos,left_pos))
        
        #slicing the faces from the image
        current_face_image= current_frame[top_pos:bottom_pos,left_pos:right_pos]
        
        #display rectangle on the face detected
        cv2.rectangle(current_frame,(left_pos,top_pos),(right_pos,bottom_pos),(0,0,255),2)
        
        #preprocess input,convert it to an image like as the data in dataset
        #convert to grayscale
        current_face_image= cv2.cvtColor(current_face_image, cv2.COLOR_BGR2GRAY)
        #resize to 48x48 px size
        current_face_image= cv2.resize(current_face_image, (48,48))
        #convert PIL image into a 3d numpy array
        img_pixels= image.img_to_array(current_face_image)
        #expand array as single row
        img_pixels= np.expand_dims(img_pixels, axis=0)
        img_pixels /=255
        
        #predictions for all 7 expressions
        exp_predictions= face_exp_model.predict(img_pixels)
        #find max indexed prediction value
        max_index= np.argmax(exp_predictions[0])
        #get corresponding label from emotions label
        emotions_label= emotions_label[max_index]
        
        #display name as text under image
        font= cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(current_frame, emotions_label, (left_pos,bottom_pos), font, 0.5, (255,255,255), 1)
       
    #showing the current face with rectangle drawn
    cv2.imshow('Webcam',current_frame)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
#release the stream and cam 
#close all opencv windows which are open
cap.release()
cv2.destroyAllWindows()