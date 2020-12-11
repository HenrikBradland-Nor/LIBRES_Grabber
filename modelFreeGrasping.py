import pcl
import numpy as np
import sys
import os
import time
from math import isnan

import matplotlib.pyplot as plt

import argparse

import pcl.pcl_visualization
from grabbing.suportFunctions import *
from grabbing.PointCloudVisualiser import visualisePointcloud, savePointcloudPCD


CONST_RHO = 10 #[mm]
CONST_D = 7 #[mm]
CONST_GRABS_PER_POINT = 3 #[-]

path_object = "pc_from_CAD/Kopp.pcd"
path_gripper = "pc_from_CAD/Gripper_plates.pcd"

pc_object = loadPointcLoud(path_object)
pc_grabber = loadPointcLoud(path_gripper)


x = []
for point in pc_grabber:
    if point[2] > 0:
        x.append(point)

pc_grabber = np.array(x, dtype=np.float32)

pc_object = resizePointCloud(pc_object, 4000)
list_of_random_points = randList(pc_object, 500)

print("Number of random points:", len(list_of_random_points))
print("Number of point in cloud:", len(pc_object))

t = time.time()
t_last = t

all_Xi = pcXi(pc_object, CONST_RHO, list_of_random_points)

L1 = []
x = []
best_points = []
jj = 0
for ii in range(len(pc_object)):
    if ii in list_of_random_points:
        zms = zeroMomentShift(all_Xi[jj], pc_object[ii])
        l1 = lengthOfVector(zms)
        L1.append(l1)
        jj += 1
        if len(best_points) < 10:
            best_points.append([pc_object[ii], zms])
        else:
            for kk in range(len(best_points)):
                if lengthOfVector(best_points[kk][1]) < l1:
                    best_points[kk] = [pc_object[ii], zms]
                    break
    else:
        L1.append(0)
    x.append(ii)


colured_pc = []
for ii in range(len(pc_object)):
    colur = rgbToColur(0x44, 0x44, 0x44)
    if L1[ii] > 3:
        colur = rgbToColur(0x01, 0x01, 0xFF)
    colured_pc.append(addColurToPoint(pc_object[ii]/30, colur))



grappber_positions = []
for point in best_points:
    grappbers = randomFingerPoseSampling(pc_grabber, point[0], CONST_GRABS_PER_POINT)
    for pose in grappbers:
        X_points = setOfPointsFromSurface(pc_object, pose, CONST_D)
        if len(X_points) > 0:


            all_Xi = pcXi(pc_object, CONST_RHO, X_points)
            all_zms = pcZeroMomentShift(all_Xi, pc_object[X_points])
            Ci = 0
            for ii in range(len(X_points)):
                Cp = LoCoMoProbFunction(pc_object[X_points[ii]], all_Xi[ii], all_zms[ii], pose)
                if not isnan(Cp):
                    Ci += Cp

            Ci /= len(X_points)
            grappber_positions.append([pose, Ci])
            print("\n\nNew grabbing pos")
            print("Size of grabber: ", len(pose))
            print("Ci-value:        ", Ci)
            print("Number of points:", len(X_points))
            print("Time used:       ", displayTime(time.time() - t))
            print("Delata time:     ", displayTime(time.time()-t_last),"\n\n")
            t_last = time.time()


print("Number of grabber positions", len(grappber_positions))


best_grabs = []
for ii in range(len(grappber_positions)):
    if len(best_grabs) < 3:
        best_grabs.append(grappber_positions[ii])
    else:
        flag = True
        for jj in range(len(best_grabs)):
            if flag and grappber_positions[ii][1] > best_grabs[jj][1]:
                best_grabs[jj] = grappber_positions[ii]
                flag = False


print("Number of best grabs", len(best_grabs))

for pos in best_grabs:
    pc = pos[0]
    for ii in range(len(pc)):
        colur = rgbToColur(0x44, 0xFF, 0xFF)
        colured_pc.append(addColurToPoint(pc[ii] / 30, colur))




pc_object = np.array(colured_pc, dtype=np.float32)

print("Time used", displayTime(time.time() - t))

plt.figure()
plt.scatter(x, L1)
plt.show()

visualisePointcloud(pc_object)