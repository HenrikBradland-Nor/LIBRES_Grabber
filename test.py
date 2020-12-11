import pcl
import numpy as np
import sys
import os
import random
import time


from grabbing.suportFunctions import *
from grabbing.PointCloudVisualiser import visualisePointcloud, savePointcloudPCD



pc = pcl.load("pc_from_CAD/Ball.pcd")


mfg = modelfreeGrabbing(pc, pc, 5)

all_pc = mfg.colurPointcloud(np.asarray(pc), mfg.rgbToColur(0x01, 0x01, 0xff))
visualisePointcloud(all_pc)


sys.exit()




path_name = "pc_from_CAD/Pen.pcd"
path_gripper = "pc_from_CAD/Gripper_plates.pcd"
cloud = pcl.load(path_name)
grabber_pc = pcl.load(path_gripper)

vg = cloud.make_voxel_grid_filter()
vg.set_leaf_size(1, 1, 1)
cloud_filterd = vg.filter()





mfg = modelfreeGrabbing(cloud_filterd, grabber_pc)


mfg.timeStamp()

all_xi = mfg.pcXi(5)
print("\nAll_Xi")
mfg.timeStamp()

zms = mfg.pcZeroMomentShift(all_xi)
print("\nZMS")
mfg.timeStamp()

mfg.L1Value()
print("\nL1")
mfg.timeStamp()

L1 = mfg.bestL1(2)
print("\nBest L1")
mfg.timeStamp()

fingerPos = mfg.randomFingerPoseSampling(2)
print("\nFingerpos ready")
mfg.timeStamp()

best_pos = mfg.Ci_Calculation()
print("\nBest fingerpos ready")
mfg.timeStamp()




visual = pcl.pcl_visualization.CloudViewing()
all_pc = np.asarray(cloud_filterd)
for ii in range(len(best_pos)):
    all_pc = np.concatenate((all_pc, np.asarray(best_pos[ii])))
x = pcl.PointCloud()
x.from_array(all_pc)
visual.ShowMonochromeCloud(x)

v = True
while v:
    v = not (visual.WasStopped())


sys.exit()




