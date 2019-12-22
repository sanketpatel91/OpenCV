#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import cv2

# #Read and Resize Image


img = cv2.imread('watch.jpg',1)
img.shape
cv2.imshow("Watch",img)
cv2.waitKey(5000)
cv2.destroyAllWindows()


img_r = cv2.resize(img,(int(img.shape[0]/2),int(img.shape[0]/2)))
cv2.imshow('ReSized Watch',img_r)
cv2.waitKey(5000)
cv2.destroyAllWindows()

# #Face detection Cascade



face_cas = cv2.CascadeClassifier("D:\Software\Anaconda\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml")
eyes_cascade = cv2.CascadeClassifier("D:\Software\Anaconda\Lib\site-packages\cv2\data\haarcascade_eye_tree_eyeglasses.xml")



jh = cv2.imread("Jahnvi.jpg",1)
#jh = cv2.resize(jh,(int(jh.shape[0]/2),int(jh.shape[0]/2)))
cv2.imshow('Jahnvi',jh)
#print(jh.shape)
cv2.waitKey(5000)
cv2.destroyAllWindows()


def detect_show(frame):
    face_cas = cv2.CascadeClassifier("D:\Software\Anaconda\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml")
    eyes_cascade = cv2.CascadeClassifier("D:\Software\Anaconda\Lib\site-packages\cv2\data\haarcascade_eye_tree_eyeglasses.xml")
    gray = cv2.cvtColor(cv2.imread(frame,1), cv2.COLOR_BGR2GRAY)
    faces = face_cas.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30))
    for (x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
        gray = cv2.ellipse(gray, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
        faceROI = gray[y:y+h,x:x+w]
        #-- In each face, detect eyes
        eyes = eyes_cascade.detectMultiScale(faceROI)
        for (x2,y2,w2,h2) in eyes:
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int(round((w2 + h2)*0.25))
            gray = cv2.circle(gray, eye_center, radius, (255, 0, 0 ), 4)
            
    cv2.imshow('Jahnvi',gray)
    cv2.waitKey(20000)
    cv2.destroyAllWindows()


detect_show("scar.jpg")



cap = cv2.VideoCapture(0)
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)
    
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cas.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30))
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    
    cv2.imshow('Live Video',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()