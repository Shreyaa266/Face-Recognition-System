# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 22:42:29 2020

@author: Shreya Jaitley
"""

#importing the required libraries
import cv2
import face_recognition

 
cap = cv2.VideoCapture(1)
# Check if the webcam is opened correctly
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

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
        #blur the sliced face and save it to the same array itself
        current_face_image=cv2.GaussianBlur(current_face_image,(99,99),30)
        #paste the blurred face into the actual image
        current_frame[top_pos:bottom_pos,left_pos:right_pos]=current_face_image
        
        #display rectangle on the face detected
        cv2.rectangle(current_frame,(left_pos,top_pos),(right_pos,bottom_pos),(0,0,255),2)
    #showing the current face with rectangle drawn
    cv2.imshow('Webcam',current_frame)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
#release the stream and cam 
#close all opencv windows which are open
cap.release()
cv2.destroyAllWindows()

        
    