import pcl
import numpy as np
import random
import time
from math import isnan, isinf

import pcl.pcl_visualization


'''    Model Free Grasping functions   '''
class modelfreeGrabbing():
    def __init__(self, object_pc, grabber_pc, res):

        self.object_pc = object_pc
        self.grabber_pc = grabber_pc

        self.octree = object_pc.make_octreeSearch(res)
        self.octree.add_points_from_input_cloud()

        self.start_time = time.time()
        self.last_opperation_time = self.start_time

        self.finger_pos = []
        self.best_finger_pos = []

        self.timestaps = []


    def dist(self, p1, p2):
        x = 0
        for i in range(3):
            x += np.power(p1[i] - p2[i], 2)
        return np.sqrt(x)


    def bestL1(self, numberOfValues = 10):
        list_of_L1 = []
        for ii, L1 in enumerate(self.L1):
            list_of_L1.append([ii, L1])
        list_of_L1.sort(key=lambda x:x[1])
        self.best_L1 = []
        for ii in range(numberOfValues):
            self.best_L1.append(list_of_L1[ii][0])
        return self.best_L1

    def L1Value(self, ZMS=None):
        if ZMS is None:
            self.L1 = []
            for zms in self.ZMS:
                L1 = np.zeros(1, dtype=np.float32)
                for n in zms:
                    L1 += np.power(n, 2)
                self.L1.append(np.sqrt(L1))
            return self.L1
        else:
            allL1 = []
            for zms in ZMS:
                L1 = np.zeros(1, dtype=np.float32)
                for n in zms:
                    L1 += np.power(n, 2)
                allL1.append(np.sqrt(L1))
            return allL1

    def lengthOfVector(self, v):
        x = 0
        for i in range(len(v)):
            x += np.power(v[i], 2)
        return np.sqrt(x)


    def Xi(self, pointcloud, centerPoint, radius):
        Xi = []
        for point in pointcloud:
            flag = True
            for ii in range(len(point)):
                if not centerPoint[ii]-radius < point[ii] < centerPoint[ii]+radius:
                    flag = False
            if flag:
                if dist(point, centerPoint) <= radius:
                    Xi.append(point)
        return np.array(Xi)

    def zeroMomentShift(self, Xi, centerPoint, partOfObject = True):
        points = np.zeros((1, 3), dtype=np.float32)

        if partOfObject:
            for xi in Xi:
                points[0][0] += self.object_pc[xi][0]
                points[0][1] += self.object_pc[xi][1]
                points[0][2] += self.object_pc[xi][2]
            return (points[0] / len(Xi)) - centerPoint
        else:
            for point in Xi:
                points[0][0] += point[0]
                points[0][1] += point[1]
                points[0][2] += point[2]
            return (points[0] / Xi.size) - centerPoint

    def pcZeroMomentShift(self, list_Xi, input_pc=None):
        self.ZMS = []
        if input_pc is None:
            for index, point in enumerate(self.object_pc):
                self.ZMS.append(self.zeroMomentShift(list_Xi[index], point))
        else:
            for index, point in enumerate(input_pc):
                self.ZMS.append(self.zeroMomentShift(list_Xi[index], point))
        return self.ZMS

    def pcXi(self, rho):
        self.all_Xi = []
        for jj in range(self.object_pc.size):
            [ind, sqdist] = self.octree.radius_search(self.object_pc[jj], rho)
            self.all_Xi.append(ind)
        return self.all_Xi

    def randomFingerPoseSampling(self, number_of_poses, center_points = None, pc_finger = None):
        if center_points is None:
            center_points = []
            for point in self.best_L1:
                center_points.append(np.asarray(self.object_pc[point]))
        if pc_finger is None:
            pc_finger = np.asarray(self.grabber_pc)
        else:
            pc_finger = np.asarray(pc_finger)

        for point in center_points:
            translation = point - self.centerPointOfPointCloud(pc_finger)
            trans_pc = self.translatePointCloud(pc_finger, translation)
            for _ in range(number_of_poses):
                rotation = [random.random()*2*np.pi, random.random()*2*np.pi, random.random()*2*np.pi]
                x = pcl.PointCloud()
                x.from_array(np.asarray(self.rotatePointCloud(trans_pc, rotation)))
                self.finger_pos.append(x)
        return self.finger_pos

    def Ci_Calculation(self, dist = 7, top_k = 3):
        for grabb in self.finger_pos:
            cp = self.centerPointOfPointCloud(grabb)
            [set_of_points, sqdist] = self.octree.radius_search(cp, dist)
            if len(set_of_points) > 0:
                Ci = 0
                for point in set_of_points:
                    Cp = self.LoCoMoProbFunction(self.object_pc[point], self.all_Xi[point], self.ZMS[point], grabb)
                    if not isnan(Cp) and not isinf(Cp):
                        Ci += Cp
                self.best_finger_pos.append([Ci, grabb])
        self.best_finger_pos.sort(key=lambda x:x[0])
        best = []
        for ii in range(top_k):
            best.append(self.best_finger_pos[ii][1])
        self.best_finger_pos = best
        return self.best_finger_pos

    def physicalLimitations(self):
        for pos in self.finger_pos:
            print("NOT DONE")



    def plainFromPointcloud(self, pc):
        pc = np.asarray(pc)
        a, b, c = 0, 0, 0
        while a==b or a==c or b==c:
            a, b, c = random.randint(0, len(pc)-1), random.randint(0, len(pc)-1), random.randint(0, len(pc)-1)

        a = pc[a]
        b = pc[b]
        c = pc[c]

        AB = np.array(b-a, dtype=np.float32)
        AC = np.array(c-a, dtype=np.float32)
        plain = np.cross(AB, AC)
        plain = np.append(plain, -(plain[0]*a[0]+plain[1]*a[1]+plain[2]+a[2]))
        return plain

    def distanceToSurface(self, plain, point):
        return np.abs(plain[0]*point[0] + plain[1]*point[1] + plain[2]*point[2] + plain[3]) / np.sqrt(plain[0]**2 + plain[1]**2 + plain[2]**2)

    def projectPointToPlain(self, point, plain):
        normalVector = np.array(plain[:3], dtype=np.float32) / np.sqrt(plain[0]**2 + plain[1]**2 + plain[2]**2)
        distance = self.distanceToSurface(plain, point)
        shiftedPoint = point + distance * normalVector
        return shiftedPoint

    def multivariateGaussianDensityFunction(self, X, my, covMat, n=3):
        dX = (X - my)
        numerator = np.exp(-0.5 * np.matmul(dX, np.matmul(np.linalg.inv(covMat), dX)))
        denominator = np.sqrt((2*np.pi)**n * np.linalg.det(covMat))
        return numerator/denominator

    def LoCoMoProbFunction(self, point, Xi, pointZMS, grabber):
        grabberPlain = self.plainFromPointcloud(grabber)
        projectionPoint = self.projectPointToPlain(point, grabberPlain)
        projectedPointZMS = self.zeroMomentShift(grabber, projectionPoint, False)
        nullVector = np.array([0, 0, 0], dtype=np.float32)

        xi_points = np.zeros((Xi.size, 3), dtype=np.float32)
        for ii in range(Xi.size):
            xi_points[ii][0] = self.object_pc[Xi[ii]][0]
            xi_points[ii][1] = self.object_pc[Xi[ii]][1]
            xi_points[ii][2] = self.object_pc[Xi[ii]][2]


        covMat = np.cov(np.transpose(xi_points))

        epsilon = pointZMS - projectedPointZMS

        mvgdf = [self.multivariateGaussianDensityFunction(pointZMS, nullVector, covMat),
                 self.multivariateGaussianDensityFunction(epsilon, nullVector, covMat)]
        if mvgdf[0]==0:
            return 0
        Cp = 1 - (mvgdf[0] - mvgdf[1])/(mvgdf[0])
        return Cp

    def physicalConstrain(self, pc, grabber):
        cornerPoints = []

        centerPoint = centerPointOfPointCloud(grabber)
        for point in grabber:
            x = 1

        return x


    '''    Point cloud manipulation functions   '''
    def pointCloudFromSet(self, pc, indicis):
        points = np.zeros((len(indicis),3), dtype=np.float32)
        for ii in range(len(indicis)):
            points[ii][0] = pc[indicis[ii]][0]
            points[ii][1] = pc[indicis[ii]][1]
            points[ii][2] = pc[indicis[ii]][2]
        return points

    def translatePointCloud(self, pc, movement):
        new_pc = []
        for point in pc:
            new_pc.append(point+movement)
        return np.array(new_pc, dtype=np.float32)

    def rotatePointCloud(self, pc, r):
        center_point = self.centerpointPointcloud(pc)
        R = np.array([[np.cos(r[0])*np.cos(r[1]),   np.cos(r[0])*np.sin(r[1])*np.sin(r[2])-np.sin(r[0])*np.cos(r[2]),   np.cos(r[0])*np.sin(r[1])*np.cos(r[2])+np.sin(r[0])*np.sin(r[2])],
                      [np.sin(r[0])*np.cos(r[1]),   np.sin(r[0])*np.sin(r[1])*np.sin(r[2])+np.cos(r[0])*np.cos(r[2]),   np.sin(r[0])*np.sin(r[1])*np.cos(r[2])-np.cos(r[0])*np.sin(r[2])],
                      [-np.sin(r[1]),               np.cos(r[1])*np.sin(r[2]),                                          np.cos(r[1])*np.cos(r[2])]])
        new_pc = []
        for point in pc:
            new_pc.append(np.matmul(R, point-center_point)+center_point)

        return np.array(new_pc, dtype=np.float32)

    def centerPointOfPointCloud(self, pc):
        p = np.zeros((1, 3), dtype=np.float32)
        for point in pc:
            p[0][0] += point[0]
            p[0][1] += point[1]
            p[0][2] += point[2]
        p = p/pc.size
        x = pcl.PointCloud()
        x.from_array(p)
        return x[0]

    def loadPointcLoud(self, path):
        p = pcl.load(path)
        return np.asarray(p)

    def centerpointPointcloud(self, pc):
        center_point = [0,0,0]
        for point in pc:
            center_point += point
        return np.array(center_point/len(pc))
    '''
    def randRemove(self, pc, ref):
        new_pc = []
        for point in pc:
            if random.random() >= ref:
                new_pc.append(point)
        return np.array(new_pc, dtype=np.float32)

    def randList(self, pc, nPoints):
        ref = 1 - nPoints / len(pc)
        new_list = []
        for point in range(len(pc)):
            if random.random() >= ref:
                new_list.append(point)
        return list(new_list)

    def scalePc(self, pc, factor):
        new_pc = []
        for point in pc:
            new_pc.append(point*factor)
        return np.array(new_pc, dtype=np.float32)


    def resizePointCloud(self, pc, nPoints):
        ref = 1-nPoints/len(pc)
        if ref < 0:
            return pc
        else:
            return randRemove(pc, ref)

    def randColurPointcloud(self, pc, color, ref):
        x = []
        for point in pc:
            c = 0x0000FF
            if random.random() >= ref:
                c = color
            x.append([point[0], point[1], point[2], c])
        return np.array(x, dtype=np.float)
    '''
    def rgbToColur(self, r, g, b):
        c = r*0x10000 + g*0x100 + b
        return c

    def addColurToPoint(self, point, colur):
        return [point[0], point[1], point[2], colur]


    def colurPointcloud(self, pc, colur):
        x = []
        for point in pc:
            x.append(self.addColurToPoint(point, colur))
        return np.array(x, dtype=np.float32)

    def pointcloudMaxMin(self, pc, aXis):
        max = -np.inf
        min = np.inf
        for point in pc:
            if point[aXis] > max:
                max = point[aXis]
            elif point[aXis] < min:
                min = point[aXis]
        dif = max - min
        return dif, max, min


    def gridSegmentation(self, pc, grid, rho):

        dx, x_max, x_min = pointcloudMaxMin(pc, 0)
        dy, y_max, y_min = pointcloudMaxMin(pc, 1)
        dz, z_max, z_min = pointcloudMaxMin(pc, 2)

        dx /= grid[0]
        dy /= grid[1]
        dz /= grid[2]

        for _x in range(grid[0]):
            x_target = x_min + _x * dx
            pc_x = [i for i in pc if x_target - rho < i[0] < x_target + dx + rho]
            for _y in range(grid[1]):
                y_target = y_min + _y * dy
                pc_y = [i for i in pc_x if y_target - rho < i[1] < y_target + dy + rho]
                for _z in range(grid[2]):
                    z_target = y_min + _z * dz
                    pc_z = np.array([i for i in pc_y if z_target - rho < i[2] < z_target + dz + rho])
                    E = pcXi(pc_z, rho)
                    return E

    def gridUniting(self, grid):
        pc = []
        for x in grid:
            for y in x:
                for z in y:
                    pc.append(z)

        return np.array(pc, dtype=np.float32)

    '''    Metric functions   '''

    def millisToH_M_S_MS(self, time):
        s = time % 60
        m = (time % 3600 - s) / 60
        h = (time - m * 60 - s) / 3600
        return str(int(h)) + "." + str(int(m)) + "." + str(int(s)) + "." + str(int(s % 1 * 1000)) + " (h.m.s.ms)"

    def timeStamp(self):

        delta_time = time.time() - self.last_opperation_time
        tot_time = time.time() - self.start_time

        self.last_opperation_time = time.time()

        self.timestaps.append(tot_time)

        print("Time since start:           ", self.millisToH_M_S_MS(tot_time))
        print("Time since last opperation: ", self.millisToH_M_S_MS(delta_time))




