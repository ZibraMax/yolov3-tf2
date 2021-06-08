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


def processFrame(f1, f2, d1=3.6, tol=0.06):
    """Modifies CVS file for frame 2 based on frame 1 

    Args:
            f1 (int): frame 1 id
            f2 (int): frame 2 id
            d1 (float, optional): Car average velocity [pix/frame]. Defaults to 3.6.
            tol (float, optional): Maximum prediction tolerancy [frame]. Defaults to 0.06.
    """

    PATH_F1 = os.path.join(INPUT_FRAMES_DIRECTORY, f'frame_{f1-1}.csv')
    PATH_F2 = os.path.join(INPUT_FRAMES_DIRECTORY, f'frame_{f2-1}.csv')

    frame1 = np.loadtxt(PATH_F1, delimiter=',', dtype='str')
    if len(frame1.shape) == 1:
        frame1 = frame1.reshape([1, 5])

    frame1 = frame1[:, :].astype(float).tolist()
    frame2 = np.loadtxt(PATH_F2, delimiter=',', dtype='str')
    if len(frame2.shape) == 1:
        frame2 = frame2.reshape([1, 5])
    frame2 = frame2[:, :].astype(float).tolist()

    for frame in [frame1, frame2]:
        for obj in frame:
            inputX1 = obj[0]
            inputY1 = obj[1]
            inputX2 = obj[2]
            inputY2 = obj[3]
            centerX = ((inputX2 - inputX1) / 2) + inputX1
            centerY = ((inputY2 - inputY1) / 2) + inputY1

            azFrame = abs(np.arctan((centerY - inputY1) /
                          (centerX - inputX1))) + np.pi/2
            pred_cx = centerX + d1*np.sin(azFrame)
            pred_cy = centerY + d1*np.cos(azFrame)
            obj += [centerX, centerY, azFrame, pred_cx, pred_cy, 'NOT FOUND']
    carros = []
    for objstart in frame1:
        carro = []
        xpred = objstart[7]
        ypred = objstart[8]
        for objend in frame2:
            xreal = objend[4]
            yreal = objend[5]
            err = np.max([abs((xreal-xpred)/xreal), abs((yreal-ypred)/yreal)])
            if err <= tol:
                objend[-1] = 'FOUND'
                _X = [objstart[4], objend[4]]
                _Y = [objstart[5], objend[5]]
                carro += [[_X, _Y]]
                # plt.plot(_X, _Y, color='black', linewidth=3)
                #vx = xreal-objstart[5]
                #vy = yreal-objstart[6]
                # objend+=[vx,vy]
                break
        carros += carro
    # for obj in frame2:
    #	if obj[9]=='NOT FOUND':
    #		obj[4] = '-1'
    #		obj+=['NF','NF']

    return carros


class Carro():
    """docstring for Carro
    """

    def __init__(self, X, Y):
        self.x = [X]
        self.y = [Y]

    def agregarPuntoTrayectoria(self, X, Y):
        self.x += [X]
        self.y += [Y]

    def ultimoPunto(self):
        return [self.x[-1], self.y[-1]]

    def trayectoria(self):
        return [[self.x[0], self.y[0]], [self.x[-1], self.y[-1]]]

    def imprimirTrayectoria(self, filename):
        np.savetxt(filename, np.array(
            [self.x, self.y]).T, delimiter=',', fmt='%s')


def seg_intersect(a1, a2, b1, b2):
    """ 
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    s = np.vstack([a1, a2, b1, b2])        # s for stacked
    h = np.hstack((s, np.ones((4, 1))))  # h for homogeneous
    l1 = np.cross(h[0], h[1])           # get first line
    l2 = np.cross(h[2], h[3])           # get second line
    x, y, z = np.cross(l1, l2)          # point of intersection
    if z == 0:                          # lines are parallel
        return (float('inf'), float('inf'))
    return (x/z, y/z)


def isBetween(p, L):
    s = False
    Xinf = min(L[0][0], L[1][0])
    Yinf = min(L[0][1], L[1][1])
    Xsup = max(L[0][0], L[1][0])
    Ysup = max(L[0][1], L[1][1])
    if p[0] > Xinf and p[0] < Xsup:
        s = True
    if p[1] > Yinf and p[1] < Ysup:
        pass
    else:
        s = False
    return s


def main():
    carros = []
    resultado = processFrame(1, 2)
    for fila in resultado:
        carros += [Carro(fila[0][0], fila[1][0])]
    for i in range(frame_count-1):
        resultado = processFrame(i+1, i+2)
        for fila in resultado:
            ci = [fila[0][0], fila[1][0]]
            cf = [fila[0][1], fila[1][1]]
            detectado = False
            for carro in carros:
                if carro.ultimoPunto() == ci:
                    detectado = True
                    carro.agregarPuntoTrayectoria(*cf)
            if not detectado:
                carros += [Carro(*cf)]
    for i, carro in enumerate(carros):
        OUTPUT_PATH = os.path.join(OUTPUT_FRAMES_DIRECTORY, f'carro_{i}.csv')
        carro.imprimirTrayectoria(OUTPUT_PATH)
    CHECKPOINTS = []
    CHECKPOINTS += [np.array([[300, 450], [600, 300]])]
    CHECKPOINTS += [np.array([[250, 450], [600, 250]])]
    for CHECKPOINT in CHECKPOINTS:
        CONTADOR = 0
        for carro in carros:
            trayectoria = np.array(carro.trayectoria())
            ppaso = seg_intersect(*trayectoria, *CHECKPOINT)
            paso = isBetween(ppaso, CHECKPOINT)*isBetween(ppaso, trayectoria)
            if paso:
                plt.plot(*np.array(carro.trayectoria()).T,
                         color='green', linewidth=3)
            else:
                plt.plot(*np.array(carro.trayectoria()).T,
                         color='red', linewidth=3)
            CONTADOR += paso
        plt.plot(*CHECKPOINT.T, '--',
                 linewidth=3, label=f'{CONTADOR} carros')
    plt.gca().invert_yaxis()
    img = plt.imread('data/frames_cars/frame_1.png')
    plt.imshow(img)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
