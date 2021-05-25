# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 22:09:34 2020

@author: Shreya Jaitley
"""

#importing the required libraries
import cv2
import face_recognition


#loading the image to detect
image_to_detect=cv2.imread('CODE/images/testing/trump-modi.jpg')

cv2.imshow("test",image_to_detect)

#detect all faces in image
all_face_locations=face_recognition.face_locations(image_to_detect,model="hog")

#print the number of faces detected
print('There are {} faces in this image'.format(len(all_face_locations)))

#looping through the face locations
for index,current_face_location in enumerate(all_face_locations):
    #splitting the tuple to get four position values
    top_pos,right_pos,bottom_pos,left_pos=current_face_location
    print('Found face {} at top:{},right:{},bottom:{},left:{}'.format(index+1,top_pos,right_pos,bottom_pos,left_pos))
    
    #slicing the faces from the image
    current_face_image=image_to_detect[top_pos:bottom_pos,left_pos:right_pos]
    cv2.imshow("Face no"+str(index+1),current_face_image)
    