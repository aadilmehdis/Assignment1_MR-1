#!/usr/bin/env python
# coding: utf-8


# Importing libraries
import numpy as np
import load_points
import matplotlib.pyplot as plt
import matplotlib as mpl



# World to Camera transformation of Frame
# 1. Translate by [27,6,-8]
# 2. Rotate by -90˚ along the Z axis
# 3. Rotate by -90˚ along the X axis

# Thus,
#     Pworld = [ 1 0 0 27 ] [ 0 1 0 0 ] [1  0 0 0] Pcamera
#              [ 0 1 0  6 ] [-1 0 0 0 ] [0  0 1 0]
#              [ 0 0 1 -8 ] [ 0 0 1 0 ] [0 -1 0 0]
#              [ 0 0 0  1 ] [ 0 0 0 1 ] [0  0 0 1]




# Debug Code V1
# -------------
# T1 = np.array([[1,0,0,-0.27],[0,1,0,-0.06],[0,0,1,0.08],[0,0,0,1]])
# R1 = np.array([[0,1,0,0],[-1,0,0,0],[0,0,1,0],[0,0,0,1]])
# R2 = np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]])
# K = np.array([[7.215377e+02, 0.000000e+00, 6.095593e+02],[0.000000e+00, 7.215377e+02, 1.728540e+02],[0.000000e+00, 0.000000e+00, 1.000000e+00]])

# Debug Code V2
# -------------
# T1 = np.array([[1,0,0,-0.27],[0,1,0,-0.06],[0,0,1,0.08],[0,0,0,1]])
# R1 = np.array([[0,-1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,1]])
# R2 = np.array([[0,0,-1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]])
# K = np.array([[7.215377e+02, 0.000000e+00, 6.095593e+02],[0.000000e+00, 7.215377e+02, 1.728540e+02],[0.000000e+00, 0.000000e+00, 1.000000e+00]])

# M = np.matmul(np.matmul(T1,R1),R2)
# M = np.matmul(np.matmul(R1,R2),T1)
# M = np.matmul(np.matmul(R1,R2),T1)
# K = np.append(K, np.zeros((K.shape[0],1)), axis=1)
# P = np.matmul(K,M)


# Current Code 

# Pushing the world origin by the Camera's position i.e. Making the Camera the origin of the world
T  = np.array([[1,0,0,-0.27],[0,1,0,-0.06],[0,0,1,0.08]])

# Rotate -90˚ along Z axis
R1 = np.array([[0,-1,0],[1,0,0],[0,0,1]])

# Rotate -90˚ along X axis
R2 = np.array([[1,0,0],[0,0,-1],[0,1,0]])

# Obtain the compound rotation
R  = np.matmul(R2,R1)

# Intrinsic Camera Calibration Matrix
K  = np.array([[7.215377e+02, 0.000000e+00, 6.095593e+02],[0.000000e+00, 7.215377e+02, 1.728540e+02],[0.000000e+00, 0.000000e+00, 1.000000e+00]])

# Getting the Projection Matrix
P = np.matmul(K,np.matmul(R,T))


# Getting the Lidar points
points = load_points.load_velodyne_points('lidar-points.bin')
# Augmenting 1 for every vector to Homogenise the Vectors
points = np.append(points, np.ones((points.shape[0],1)), axis=1)

# Projecting Lidar points onto the image plane
image_points = np.matmul(P, points.T)
image_points = image_points.T

pic = plt.imread('image.png')
# Plotting Image Points
image_points[:,0] /= image_points[:,2]
image_points[:,1] /= image_points[:,2]
image_points[:,2] /= image_points[:,2]
plt.imshow(pic)
print(np.min(points[:,2]))
print(np.max(points[:,2]))
norm = mpl.colors.Normalize(vmin = np.min(points[:,2]),vmax = np.max(points[:,2]))
mapcolor = ['red','yellow','blue','green']
plt.scatter(image_points[:,0], image_points[:,1],s =0.2,c = points[:,2]*1000,cmap= mpl.cm.get_cmap('gist_ncar'))


plt.ylim((0, 375))
plt.xlim((0, 1242))
plt.gca().invert_yaxis()
plt.show()









