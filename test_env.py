# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 13:23:04 2020

@author: Shreya Jaitley
"""

import cv2
import dlib
import face_recognition

print(cv2.__version__)
print(dlib.__version__)
print(face_recognition.__version__)


#loading image to detect
image_test=cv2.imread('CODE/images/testing/trump-modi.jpg')

#display image
cv2.imshow("Image",image_test)