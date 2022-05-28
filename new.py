import cv2
import time
import mediapipe as mp
import argparse
import pandas as pd
import os




mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils
points = mpPose.PoseLandmark
path = "Yoga_POSE//"


data = []
for p in points:
        x = str(p)[13:]
        data.append(x + "_x")
        data.append(x + "_y")
        data.append(x + "_z")
        data.append(x + "_vis")

data = pd.DataFrame(columns = data) # Empty dataset


def video_capture(mpath):
	for folder in os.listdir(mpath):
		fpath = mpath+folder
		print("folderpath",fpath)
		increment = 0
		for file in os.listdir(fpath):
			path = os.path.join(fpath,file)
			print("filepath",path)
			cap = cv2.VideoCapture(path)
 
			pTime,count=0,0

			while(cap.isOpened()):
				success, img = cap.read()
				if (success != True):
					break
				imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
				results = pose.process(imgRGB)
				
				temp=[]

				if results.pose_landmarks:
					mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
					landmarks = results.pose_landmarks.landmark

					for i,j in zip(points,landmarks):
						temp = temp + [j.x, j.y, j.z, j.visibility]

					data.loc[count] = temp
				count +=1

				cTime = time.time()
				fps = 1 / (cTime - pTime)
				pTime = cTime

				cv2.putText(img, str(int(fps)),(70,58),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
				cv2.imshow("Image", img)
				cv2.waitKey(1)
			

		data['label']= 'Vrikshasana'
		data.to_csv("csv/Vrikshasana.csv")
		increment +=1
		cap.release()


# Empty dataset
video_capture(path)






