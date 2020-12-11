import pcl
import numpy as np
import sys
import os
import time
from math import isnan, isinf

import matplotlib.pyplot as plt

import argparse

import pcl.pcl_visualization
from grabbing.suportFunctions import *
from grabbing.PointCloudVisualiser import visualisePointcloud, savePointcloudPCD

CONST_RHO = 5 #[mm] 4
CONST_D = .3 #[mm] 0.1
RESELUTION = 5 #[-]
CONST_GRABS_PER_POINT = 3 #[-]
CONST_NUMBER_OF_POINTS_TO_EVALUATE = 20 #[-]

flagg = True

stepp = 0.01
learning_rate = 0.00001

value = 2.91746749663353
delta = 1000
all_delta = []
all_value = []

all_value = [4, 3.810628833770752, 3.754530492782593, 3.6987025270462035, 3.637948455810547, 3.5822387940883638, 3.3860499851703647, 3.2274392061233526, 3.070329869985581, 2.91746749663353]




def run(a):
    CONST_RHO = a

    print("\n\n\n"
          "new run "
          "\n"
          "parameter:", a, "\n\n")

    t = time.time()

    path_object = "pc_from_CAD/Kopp.pcd"
    path_gripper = "pc_from_CAD/Gripper_plates.pcd"

    pc_object = pcl.load(path_object)
    pc_grabber = pcl.load(path_gripper)



    #print("Initial number of points:", pc_object.size)
    vg = pc_object.make_voxel_grid_filter()
    vg.set_leaf_size(1, 1, 1)
    pc_object_reduced = vg.filter()
    #print("Reduced number of points:", pc_object_reduced.size)


    mfg = modelfreeGrabbing(pc_object_reduced, pc_grabber, res=RESELUTION)
    #print("\nOcTree created.")
    mfg.timeStamp()



    all_Xi = mfg.pcXi(CONST_RHO)
    #print("\nAll Xi-sets calculated.")
    #print(len(all_Xi[0]))
    mfg.timeStamp()


    zms = mfg.pcZeroMomentShift(all_Xi)
    #print("\nAll ZMS calculated.")
    mfg.timeStamp()


    L1 = mfg.L1Value()
    #print("\nAll L1-values calculated.")
    mfg.timeStamp()


    bestL1 = mfg.bestL1(CONST_NUMBER_OF_POINTS_TO_EVALUATE)
    #print("\nAll L1-values sorted and best L1-values are sorted out")
    mfg.timeStamp()


    fingerPos = mfg.randomFingerPoseSampling(CONST_GRABS_PER_POINT)
    #print("\nRandom fingerposes created")
    mfg.timeStamp()


    best_fingerPos = mfg.Ci_Calculation(top_k=1)
    #print("\nBest fingerpos ready")
    mfg.timeStamp()

    new_t = time.time()
    print(os.getcwd())
    f = open("log.txt", "a")
    f.write("\n" + path_object)
    f.write(", " + str(pc_object_reduced.size))
    f.write(", " + str(CONST_RHO))
    f.write(", " + str(len(all_Xi[0])))
    f.write(", " + str(CONST_D))
    f.write(", " + str(RESELUTION))
    f.write(", " + str(CONST_GRABS_PER_POINT*CONST_NUMBER_OF_POINTS_TO_EVALUATE))

    for t in mfg.timestaps:
        f.write(", " + str(t))
    f.close()

    return new_t - t

def update(x, stepp, lr):
    h = run(x)
    h_dt = run(x + stepp)
    dh = (h - h_dt)/stepp
    x = x + dh*lr

    return dh, x


for _ in range(5):
    delta, value = update(value, stepp, learning_rate)
    all_delta.append(delta)
    all_value.append(value)


print(all_value)
x = []
for ii in range(len(all_value)):
    x.append(ii)
plt.figure()
plt.plot(x, all_value)
plt.show()



#all_pc = mfg.colurPointcloud(np.asarray(pc_object_reduced), mfg.rgbToColur(0x01, 0x01, 0xff))
#for ii in range(len(best_fingerPos)):
#    all_pc = np.concatenate((all_pc, mfg.colurPointcloud(np.asarray(best_fingerPos[ii]), mfg.rgbToColur(127, 127, 127))))


visualisePointcloud(all_pc)