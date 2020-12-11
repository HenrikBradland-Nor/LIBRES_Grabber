import pcl
import numpy as np
import sys
import os

import pcl.pcl_visualization
import grabbing.suportFunctions as sf



def visualisePointcloud(pc):
    if len(pc[0]) == 3:
        p = pcl.PointCloud()
        p.from_array(pc)
        visual = pcl.pcl_visualization.CloudViewing()
        visual.ShowMonochromeCloud(p)
    else:
        p = pcl.PointCloud_PointXYZRGB()
        p.from_array(pc)
        visual = pcl.pcl_visualization.CloudViewing()
        visual.ShowColorCloud(p)

    v = True
    while v:
        v = not (visual.WasStopped())

def savePointcloudPCD(pc, file):
    if len(pc[0]) == 3:
        header = "# .PCD v.7 - Point Cloud Data file format\n" \
                 "VERSION .7\n" \
                 "FIELDS x y z\n" \
                 "SIZE 4 4 4\n" \
                 "TYPE F F F \n" \
                 "COUNT 1 1 1\n" \
                 "WIDTH XXXX\n" \
                 "HEIGHT 1\n" \
                 "VIEWPOINT 0 0 0 1 0 0 0\n" \
                 "POINTS XXXX\n" \
                 "DATA ascii\n"
    else:
        header = "# .PCD v.7 - Point Cloud Data file format\n" \
                 "VERSION .7\n" \
                 "FIELDS x y z rgb\n" \
                 "SIZE 4 4 4 4\n" \
                 "TYPE F F F F\n" \
                 "COUNT 1 1 1 1\n" \
                 "WIDTH XXXX\n" \
                 "HEIGHT 1\n" \
                 "VIEWPOINT 0 0 0 1 0 0 0\n" \
                 "POINTS XXXX\n" \
                 "DATA ascii\n"

    header = header.replace("XXXX", str(len(pc)))

    f = open(file, "w")
    f.write(header)
    for point in pc:
        for p in point:
            f.write(str(p) + " ")
        f.write("\n")

    f.close()



'''

ROH = 0.01



path_name = "pc_from_CAD/Test.pcd"
p = pcl.load_XYZRGB(path_name)


filtering = False


seg = p.make_segmenter()

seg.set_optimize_coefficients(True)

seg.set_model_type(pcl.SACMODEL_PLANE)
seg.set_method_type(pcl.SAC_RANSAC)
seg.set_distance_threshold(0.4)

inliers, model = seg.segment()

passthrough = p.make_passthrough_filter()
passthrough.set_filter_field_name("z")
passthrough.set_filter_limits(0.0, 1600)

cloud_filtered = passthrough.filter()

for x in range(len(cloud)):
    cloud[x][2] *= 4
    if cloud[x][2] > 0:
        cloud[x][2] = 0

#print(cloud)
cloud = p - np.mean(p, 0)

x = []
for point in cloud:
    if point[1] < 0:
        x.append(point)
x= np.array(x, dtype=np.float32)

#print("Inliers", "%.4f" % (100*len(inliers)/len(cloud)),"%")

ptcloud_centred = pcl.PointCloud()

pcl.PointCloud_PointXYZRGB()

if filtering:
    ptcloud_centred.from_array(cloud[inliers])
#else:
    #ptcloud_centred.
    #ptcloud_centred.from_array(cloud_filtered)
visual = pcl.pcl_visualization.CloudViewing()

#print(cloud[inliers])



visual.ShowColorCloud(p)

v = True

while v:
    v = not(visual.WasStopped())

'''