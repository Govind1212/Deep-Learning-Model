import cv2
import time
import mediapipe as mp
import argparse
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

import joblib
filename = 'finalized_model.sav'
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils
points = mpPose.PoseLandmark

data = []
for p in points:
        x = str(p)[13:]
        data.append(x + "_x")
        data.append(x + "_y")
        data.append(x + "_z")
        data.append(x + "_vis")

data = pd.DataFrame(columns = data)
# construct the argument parser and parse the arguments
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("Yoga_POSE/Vrikshasana/Abhay_Vriksh.mp4")

pTime,count=0,0

while(cap.isOpened()):
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    temp=[]
    if results.pose_landmarks:
    	mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    	landmarks = results.pose_landmarks.landmark
    	for i,j in zip(points,landmarks):
    		temp = temp + [j.x, j.y, j.z, j.visibility]
    	data.loc[count] = temp
    loaded_model = joblib.load(filename)
    pred=loaded_model.predict(data)
    for p in pred:
    	if   p==0:
    		print("Padmasana")
    	elif p==1:
    		print("Tadasana")

    	elif p==2:
    		print("Trikonasana")

    	elif p==3:
    		print("Vrikshasana")

    	else:
    		print("None")

    count +=1


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)),(70,58),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
cap.release()
