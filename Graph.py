import numpy as np
import matplotlib.pyplot as plt
import os

INPUT_FRAMES_DIRECTORY1 = 'data/OBJS'
INPUT_FRAMES_DIRECTORY2 = 'data/Output_frames/'

frames = os.listdir(INPUT_FRAMES_DIRECTORY1)
framesorig = os.listdir(INPUT_FRAMES_DIRECTORY2)
for i in range(1,len(frames)):

	PATH_FIN = os.path.join(INPUT_FRAMES_DIRECTORY1, f'obj_{i+1}.csv')
	PATH_ORIG = os.path.join(INPUT_FRAMES_DIRECTORY2, f'frame_{i}.csv')

	frame1 = np.loadtxt(PATH_FIN,delimiter=',',dtype='str')
	if len(frame1.shape)==1:
		frame1 = frame1.reshape([1,len(frame1)])
	frame1 = frame1[:,:-1].astype(float).tolist()
	frame2 = np.loadtxt(PATH_ORIG,delimiter=',',dtype='str')
	if len(frame2.shape)==1:
		frame2 = frame2.reshape([1,len(frame2)])
	frame2 = frame2[:,:-1].astype(float).tolist()

	for obj in frame1:
		if obj[10]=='FOUND':
			for objfin in frame2:
				pass #encontrar objeto final y llamarlo obj_fin
			cx_orig = 0
			cy_orig = 0
			cx_fin = 0
			cy_fin = 0

