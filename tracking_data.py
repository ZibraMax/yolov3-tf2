#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

INPUT_FRAMES_DIRECTORY = 'data/Output_frames/'
OUTPUT_FRAMES_DIRECTORY = 'data/OBJS'
try:
	os.makedirs(OUTPUT_FRAMES_DIRECTORY)
except:
	pass

frame_count = len(os.listdir(INPUT_FRAMES_DIRECTORY))


def processFrame(f1,f2,d1=3.6,tol=0.032):
	"""Modifies CVS file for frame 2 based on frame 1 

	Args:
		f1 (int): frame 1 id
		f2 (int): frame 2 id
		d1 (float, optional): Car average velocity [pix/frame]. Defaults to 3.6.
		tol (float, optional): Maximum prediction tolerancy [frame]. Defaults to 0.032.
	"""

	PATH_F1 = os.path.join(INPUT_FRAMES_DIRECTORY, f'frame_{f1-1}.csv')
	PATH_F2 = os.path.join(INPUT_FRAMES_DIRECTORY, f'frame_{f2-1}.csv')

	frame1 = np.loadtxt(PATH_F1,delimiter=',',dtype='str')
	if len(frame1.shape)==1:
		frame1 = frame1.reshape([1,4])
	frame1 = frame1[:,:].astype(float).tolist()
	frame2 = np.loadtxt(PATH_F2,delimiter=',',dtype='str')
	if len(frame2.shape)==1:
		frame2 = frame2.reshape([1,4])
	frame2 = frame2[:,:].astype(float).tolist()

	for frame in [frame1,frame2]:
		for obj in frame:
			inputX1 = obj[0]
			inputY1 = obj[1]
			inputX2 = obj[2]
			inputY2 = obj[3]
			centerX = ((inputX2 - inputX1) / 2) + inputX1
			centerY = ((inputY2 - inputY1) / 2) + inputY1
			#x = int(centerX - (inputW / 2))
			#y = int(centerY + (inputH / 2))

			azFrame = abs(np.arctan((centerY - inputY1)/(centerX - inputX1)))+ np.pi/2
			pred_cx = centerX + d1*np.sin(azFrame)
			pred_cy = centerY + d1*np.cos(azFrame)
			obj+=[centerX,centerY,azFrame,pred_cx,pred_cy,'NOT FOUND']

	for objstart in frame1:
		xpred = objstart[7]
		ypred = objstart[8]
		for objend in frame2:
			xreal = objend[4]
			yreal = objend[5]
			err = np.max([abs((xreal-xpred)/xreal),abs((yreal-ypred)/yreal)])
			if err <= tol:
				#objend[4]=objstart[4]
				objend[-1] = 'FOUND'
				_X = [objstart[5],objend[5]]
				_Y = [objstart[6],objend[6]]
				plt.plot(_X,_Y,color='black',linewidth=3)
				vx = xreal-objstart[5]
				vy = yreal-objstart[6]
				objend+=[vx,vy]
				break
	#for obj in frame2:
	#	if obj[9]=='NOT FOUND':
	#		obj[4] = '-1'
	#		obj+=['NF','NF']

	OUTPUT_PATH = os.path.join(OUTPUT_FRAMES_DIRECTORY,f'obj_{f2}.csv')
	np.savetxt(OUTPUT_PATH,frame2,delimiter=',',fmt='%s')

def main():
	for i in range(frame_count-1):
		processFrame(i+1,i+2)
	plt.gca().invert_yaxis()
	img = plt.imread('data/frames_cars/frame_1.png')
	plt.imshow(img)
	# 
	plt.show()

if __name__ == '__main__':
	main()



# for f in range(0,len(dirListing)-1 ):

# 	#Reading input files

# 	route1 = 'data_outputs/csv/frame_'+ str(f+1) + '.csv'
# 	route2 = 'data_outputs/csv/frame_'+ str(f+2) + '.csv'

# 	print(route1)

# 	frame1 = pd.read_csv(route1, header = None).to_numpy()
# 	frame2 = pd.read_csv(route2, header = None).to_numpy()

# 	objs_f1 = []
# 	objs_f2 = []
# 	resp1 = np.zeros(frame1.shape[0])
# 	resp2 = np.zeros(frame2.shape[0])

# 	for i in range(0,frame1.shape[0]):
# 		obji = frame1[i]
# 		obj_track = np.array(obji[0:])

# 		inputX = obj_track[0]
# 		inputY = obj_track[1]
# 		inputW = obj_track[2]
# 		inputH = obj_track[3]
# 		centerX = inputX + (inputW / 2)
# 		centerY = inputY + (inputH / 2)
# 		x = int(centerX - (inputW / 2))
# 		y = int(centerY + (inputH / 2))

# 		azFrame = abs(np.arctan((centerY - y)/(centerX - x)))+ np.pi/2

# 		pred_cx = centerX + 3.6*np.sin(azFrame)
# 		pred_cy = centerY + 3.6*np.cos(azFrame)

# 		resp1 = np.append(obj_track,[centerX,centerY,azFrame,pred_cx,pred_cy])
# 		objs_f1.append(resp1)

# 	for i in range(0,frame2.shape[0]):
# 		obji = frame2[i]
# 		obj_track = np.array(obji[0:])

# 		inputX = obj_track[0]
# 		inputY = obj_track[1]
# 		inputW = obj_track[2]
# 		inputH = obj_track[3]
# 		centerX = inputX + (inputW / 2)
# 		centerY = inputY + (inputH / 2)
# 		x = int(centerX + (inputW / 2))
# 		y = int(centerY + (inputH / 2))

# 		azFrame = abs(np.arctan((centerY - y)/(centerX - x))) + np.pi/2

# 		pred_cx = centerX + 3.6*np.sin(azFrame)
# 		pred_cy = centerY + 3.6*np.cos(azFrame)

# 		resp2 = np.append(obj_track,[centerX,centerY,azFrame,pred_cx,pred_cy])
# 		objs_f2.append(resp2)


# 	for i in range(0,len(objs_f1)):
# 		xpred_f2 = objs_f1[i][9]
		
# 		for j in range(0,len(objs_f2)):
# 			x_f2 = objs_f2[j][6]
# 			if xpred_f2 < x_f2*1.032 and xpred_f2 > x_f2*0.968:
# 				id_obj = objs_f1[i][4]
# 				objs_f2[j][4]=id_obj

# 	np.savetxt('objs'+str(i+1)+'.csv',objs_f1,delimiter=',',fmt='%s')
# 	np.savetxt('objs'+str(i+2)+'.csv',objs_f2,delimiter=',',fmt='%s')