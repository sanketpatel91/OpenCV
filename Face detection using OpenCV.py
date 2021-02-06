#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import cv2
from skimage import io
from skimage.transform import resize

# Image URL
url = 'https://user-images.githubusercontent.com/5509837/100316790-d1f8e080-2f80-11eb-8b98-7549e16f3556.jpg'
image = io.imread (url)
# print(image.shape)
# image = resize(image,(image.shape[0] // 4, image.shape[1] // 4,image.shape[2]),anti_aliasing=True)
# print(image.shape)
cv2.namedWindow("Image", cv2.WINDOW_NORMAL) 
cv2.imshow("Image",image)
cv2.waitKey(2000)
cv2.destroyAllWindows()

print("Face Detection in url image")

# #Face detection Cascade
#print(os.path.dirname(os.path.abspath(cv2.__file__)))
cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
haar_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')
haar_eye_model = os.path.join(cv2_base_dir, 'data/haarcascade_eye_tree_eyeglasses.xml')

face_cas = cv2.CascadeClassifier(haar_model)
eyes_cascade = cv2.CascadeClassifier(haar_eye_model)

def detect_show(frame):
    global face_cas 
    global eyes_cascade 
    # gray = cv2.cvtColor(cv2.imread(frame,1), cv2.COLOR_BGR2GRAY)
    gray = frame
    # cv2.cvtcolor (image, cv2.color_bgr2rgb)
    faces = face_cas.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30))
    for (x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
        frame = cv2.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 2)
        faceROI = frame[y:y+h,x:x+w]
        #-- In each face, detect eyes
        eyes = eyes_cascade.detectMultiScale(faceROI)
        for (x2,y2,w2,h2) in eyes:
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int(round((w2 + h2)*0.25))
            frame = cv2.circle(frame, eye_center, radius, (255, 0, 0 ), 2)

    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Image",frame)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()

detect_show(image)

import win32api
import win32con
result = win32api.MessageBox(None,'Do You want to see Live Video Face Detection?\nPress \'q\' to close window','Confirm to see Live video',win32con.MB_YESNO)

if result == win32con.IDYES:
    print('Pressed Yes')
    print("--- Live Video Face detection ---\n Wait till window open \n Press 'q' to quit window")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened:
        print('--(!)Error opening video capture')
        exit(0)
        
    # Capture frame-by-frame
    ret, frame = cap.read()
        
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cas.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30))
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        
    cv2.imshow("Image",frame)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()
        # if cv2.waitKey(1) & 0xFF == ord('q'):
            # break    
    cap.release()
    cv2.destroyAllWindows()
else:
    print('Pressed No')