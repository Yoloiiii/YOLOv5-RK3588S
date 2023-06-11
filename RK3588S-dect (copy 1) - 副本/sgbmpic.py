import cv2
import numpy as np
import time
import random
import math

# -----------------------------------双目相机的基本参数---------------------------------------------------------
#   left_camera_matrix          左相机的内参矩阵
#   right_camera_matrix         右相机的内参矩阵
#
#   left_distortion             左相机的畸变系数    格式(K1,K2,P1,P2,0)
#   right_distortion            右相机的畸变系数
# -------------------------------------------------------------------------------------------------------------
# 左镜头的内参，如焦距
left_camera_matrix = np.array([[367.5609,0.4985,342.9465],[0,367.4033,250.7901],[0.,0.,1.]])
right_camera_matrix = np.array([[369.1337,0.5095,350.0777],[0,369.0321,242.9954],[0.,0.,1.]])

# 畸变系数,K1、K2、K3为径向畸变,P1、P2为切向畸变
left_distortion = np.array([[0.0021,0.0078,-0.0012,0.0025,0]])
right_distortion = np.array([[0.0108,-0.0061,-2.0013e-05,0.0019,0]])

# 旋转矩阵
R = np.array([[0.9998,-0.0069,0.0206],
              [0.0071,0.9999,-0.0114],
              [-0.0206,0.0115,0.997]])
# 平移矩阵
T = np.array([-55.3297,-2.0746,6.1968])

size = (1280, 960)

R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion, size, R,
                                                                  T)

# 校正查找映射表,将原始图像和校正后的图像上的点一一对应起来
left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)
#print(Q)

# --------------------------鼠标回调函数---------------------------------------------------------
#   event               鼠标事件
#   param               输入参数
# -----------------------------------------------------------------------------------------------


def out_depth(x, y, param):
    threeD = param
    print('\n像素坐标 x = %d, y = %d' % (x, y))
    #print("世界坐标是：", int(threeD[y][x][0]), int(threeD[y][x][1]), int(threeD[y][x][2]), "mm")
    print("世界坐标xyz 是：", threeD[int(y)][int(x)][0]/1000, threeD[int(y)][int(x)][1]/1000, threeD[int(y)][int(x)][2]/1000, "m")

    distance = math.sqrt(threeD[int(y)][int(x)][0] ** 2 + threeD[int(y)][int(x)][1] ** 2 + threeD[int(y)][int(x)][2] ** 2)
    distance = distance / 1000.0  # mm -> m
    print("距离是：", distance, "m")




