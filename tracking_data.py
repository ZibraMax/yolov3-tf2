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


def processFrame(f1,f2,d1=3.6,tol=0.04):
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
		frame1 = frame1.reshape([1,5])
	frame1 = frame1[:,:].astype(float).tolist()
	frame2 = np.loadtxt(PATH_F2,delimiter=',',dtype='str')
	if len(frame2.shape)==1:
		frame2 = frame2.reshape([1,5])
	frame2 = frame2[:,:].astype(float).tolist()

	for frame in [frame1,frame2]:
		for obj in frame:
			inputX1 = obj[0]
			inputY1 = obj[1]
			inputX2 = obj[2]
			inputY2 = obj[3]
			centerX = ((inputX2 - inputX1) / 2) + inputX1
			centerY = ((inputY2 - inputY1) / 2) + inputY1

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
				objend[-1] = 'FOUND'
				_X = [objstart[4],objend[4]]
				_Y = [objstart[5],objend[5]]
				plt.plot(_X,_Y,color='black',linewidth=3)
				#vx = xreal-objstart[5]
				#vy = yreal-objstart[6]
				#objend+=[vx,vy]
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