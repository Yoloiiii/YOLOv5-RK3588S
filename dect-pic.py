import os
import urllib
import traceback
import time
import datetime as dt
import sys
import numpy as np
import cv2
from rknnlite.api import RKNNLite
import math
from PIL import Image
 
#模型文件 
RKNN_MODEL = 'best.rknn'
DATASET = './dataset.txt'
 
QUANTIZE_ON = True
 
OBJ_THRESH = 0.25
NMS_THRESH = 0.45
IMG_SIZE = 640
#类别种类
CLASSES = ("gemma", "ripe", "pollinated")
 
"""CLASSES = ("person", "bicycle", "car", "motorbike ", "aeroplane ", "bus ", "train", "truck ", "boat", "traffic light",
           "fire hydrant", "stop sign ", "parking meter", "bench", "bird", "cat", "dog ", "horse ", "sheep", "cow", "elephant",
           "bear", "zebra ", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
           "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife ",
           "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza ", "donut", "cake", "chair", "sofa",
           "pottedplant", "bed", "diningtable", "toilet ", "tvmonitor", "laptop	", "mouse	", "remote ", "keyboard ", "cell phone", "microwave ",
           "oven ", "toaster", "sink", "refrigerator ", "book", "clock", "vase", "scissors ", "teddy bear ", "hair drier", "toothbrush ")
""" 
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






def sigmoid(x):
    return 1 / (1 + np.exp(-x))
 
 
def xywh2xyxy(x):
    # Convert [x, y, w, h] to [x1, y1, x2, y2]
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y
 
 
def process(input, mask, anchors):
 
    anchors = [anchors[i] for i in mask]
    grid_h, grid_w = map(int, input.shape[0:2])
 
    box_confidence = sigmoid(input[..., 4])
    box_confidence = np.expand_dims(box_confidence, axis=-1)
 
    box_class_probs = sigmoid(input[..., 5:])
 
    box_xy = sigmoid(input[..., :2])*2 - 0.5
 
    col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)
    col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    grid = np.concatenate((col, row), axis=-1)
    box_xy += grid
    box_xy *= int(IMG_SIZE/grid_h)
 
    box_wh = pow(sigmoid(input[..., 2:4])*2, 2)
    box_wh = box_wh * anchors
 
    box = np.concatenate((box_xy, box_wh), axis=-1)
 
    return box, box_confidence, box_class_probs
 
 
def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with box threshold. It's a bit different with origin yolov5 post process!
    # Arguments
        boxes: ndarray, boxes of objects.
        box_confidences: ndarray, confidences of objects.
        box_class_probs: ndarray, class_probs of objects.
    # Returns
        boxes: ndarray, filtered boxes.
        classes: ndarray, classes for boxes.
        scores: ndarray, scores for boxes.
    """
    boxes = boxes.reshape(-1, 4)
    box_confidences = box_confidences.reshape(-1)
    box_class_probs = box_class_probs.reshape(-1, box_class_probs.shape[-1])
 
    _box_pos = np.where(box_confidences >= OBJ_THRESH)
    boxes = boxes[_box_pos]
    box_confidences = box_confidences[_box_pos]
    box_class_probs = box_class_probs[_box_pos]
 
    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)
    _class_pos = np.where(class_max_score >= OBJ_THRESH)
 
    boxes = boxes[_class_pos]
    classes = classes[_class_pos]
    scores = (class_max_score* box_confidences)[_class_pos]
 
    return boxes, classes, scores
 
 
def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.
    # Arguments
        boxes: ndarray, boxes of objects.
        scores: ndarray, scores of objects.
    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
 
    areas = w * h
    order = scores.argsort()[::-1]
 
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
 
        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])
 
        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1
 
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep
 
 
def yolov5_post_process(input_data):
    masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
               [59, 119], [116, 90], [156, 198], [373, 326]]
 
    boxes, classes, scores = [], [], []
    for input, mask in zip(input_data, masks):
        b, c, s = process(input, mask, anchors)
        b, c, s = filter_boxes(b, c, s)
        boxes.append(b)
        classes.append(c)
        scores.append(s)
 
    boxes = np.concatenate(boxes)
    boxes = xywh2xyxy(boxes)
    classes = np.concatenate(classes)
    scores = np.concatenate(scores)
 
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
 
        keep = nms_boxes(b, s)
 
        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])
 
    if not nclasses and not nscores:
        return None, None, None
 
    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)
 
    return boxes, classes, scores
 
 
def draw(imgL, boxes, scores, classes, times, imgR):
    """Draw the boxes on the image.
    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        fps: int.
        all_classes: all classes name.
    """

    #加载图片
    frame1 = imgL
    frame2 = imgR
    # 将BGR格式转换成灰度图片，用于畸变矫正
    imgL = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # 重映射，就是把一幅图像中某位置的像素放置到另一个图片指定位置的过程。
    # 依据MATLAB测量数据重建无畸变图片,输入图片要求为灰度图
    img1_rectified = cv2.remap(imgL, left_map1, left_map2, cv2.INTER_LINEAR)
    img2_rectified = cv2.remap(imgR, right_map1, right_map2, cv2.INTER_LINEAR)

    # 转换为opencv的BGR格式
    imageL = cv2.cvtColor(img1_rectified, cv2.COLOR_GRAY2BGR)
    imageR = cv2.cvtColor(img2_rectified, cv2.COLOR_GRAY2BGR)

        # ------------------------------------SGBM算法----------------------------------------------------------
        #   blockSize                   深度图成块，blocksize越低，其深度图就越零碎，0<blockSize<10
        #   img_channels                BGR图像的颜色通道，img_channels=3，不可更改
        #   numDisparities              SGBM感知的范围，越大生成的精度越好，速度越慢，需要被16整除，如numDisparities
        #                               取16、32、48、64等
        #   mode                        sgbm算法选择模式，以速度由快到慢为：STEREO_SGBM_MODE_SGBM_3WAY、
        #                               STEREO_SGBM_MODE_HH4、STEREO_SGBM_MODE_SGBM、STEREO_SGBM_MODE_HH。精度反之
        # ------------------------------------------------------------------------------------------------------
    blockSize = 4
    img_channels = 3
    stereo = cv2.StereoSGBM_create(minDisparity=1,
                                    numDisparities=64,
                                    blockSize=blockSize,
                                    P1=8 * img_channels * blockSize * blockSize,
                                    P2=32 * img_channels * blockSize * blockSize,
                                    disp12MaxDiff=-1,
                                    preFilterCap=1,
                                    uniquenessRatio=10,
                                    speckleWindowSize=100,
                                    speckleRange=100,
                                    mode=cv2.STEREO_SGBM_MODE_HH)
    # 计算视差
    disparity = stereo.compute(img1_rectified, img2_rectified)

    # 归一化函数算法，生成深度图（灰度图）
    disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 生成深度图（颜色图）
    dis_color = disparity
    dis_color = cv2.normalize(dis_color, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    dis_color = cv2.applyColorMap(dis_color, 2)

    # 计算三维坐标数据值
    threeD = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=True)
    # 计算出的threeD，需要乘以16，才等于现实中的距离
    threeD = threeD * 16


    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = box
        print('class: {}, score: {}'.format(CLASSES[cl], score))
        print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))
        top = int(top)
        left = int(left)
        right = int(right)
        bottom = int(bottom)
        #   空间坐标
        y = (right+top)/2
        x = (left+bottom)/2
        print('像素坐标 x = %d, y = %d' % (x, y))
        #print("世界坐标是：", int(threeD[y][x][0]), int(threeD[y][x][1]), int(threeD[y][x][2]), "mm")
        print("坐标xyz 是：", threeD[int(y)][int(x)][0]/1000, threeD[int(y)][int(x)][1]/1000, threeD[int(y)][int(x)][2]/1000, "m")

        distance = math.sqrt(threeD[int(y)][int(x)][0] ** 2 + threeD[int(y)][int(x)][1] ** 2 + threeD[int(y)][int(x)][2] ** 2)
        distance = distance / 1000.0  # mm -> m
        print("距离是：", distance, "m\n")

        #   画框
        cv2.rectangle(frame1, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(frame1, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)
 
def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
 
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
 
    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
 
    dw /= 2  # divide padding into 2 sides
    dh /= 2
 
    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)
 
# ==================================
# 如下为改动部分，主要就是去掉了官方 demo 中的模型转换代码，直接加载 rknn 模型，并将 RKNN 类换成了 rknn_toolkit2_lite 中的 RKNNLite 类
# ==================================
 
rknn = RKNNLite()
 
# load RKNN model
print('--> Load RKNN model')
ret = rknn.load_rknn(RKNN_MODEL)
 
# Init runtime environment
print('--> Init runtime environment')
# use NPU core 0 1 2
ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
if ret != 0:
    print('Init runtime environment failed!')
    exit(ret)
print('done')
 
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
#cap = cv2.VideoCapture("5.mp4")
# cap = cv2.VideoCapture(0)
 
# Check if camera opened successfully
#if (cap.isOpened()== False): 
#  print("Error opening video stream or file")
# 图片路径
img = cv2.imread('imgs/4.jpg', cv2.IMREAD_COLOR)
imgL = img[0:960, 0:1280]
imgR = img[0:960, 1280:2560]
img = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
 
# Inference
print('--> Running model')
outputs = rknn.inference(inputs=[img])
print('done')
 
# post process
input0_data = outputs[0]
input1_data = outputs[1]
input2_data = outputs[2]
 
input0_data = input0_data.reshape([3, -1]+list(input0_data.shape[-2:]))
input1_data = input1_data.reshape([3, -1]+list(input1_data.shape[-2:]))
input2_data = input2_data.reshape([3, -1]+list(input2_data.shape[-2:]))
 
input_data = list()
input_data.append(np.transpose(input0_data, (2, 3, 0, 1)))
input_data.append(np.transpose(input1_data, (2, 3, 0, 1)))
input_data.append(np.transpose(input2_data, (2, 3, 0, 1)))
 
boxes, classes, scores = yolov5_post_process(input_data)
#duration = dt.datetime.utcnow() - start
#times = round(duration.microseconds)
times = 1
# draw process result and fps
img_1 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.putText(img_1, f'time: {times}',
            (20, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 125, 125), 2)
if boxes is not None:
    draw(img_1, boxes, scores, classes, times, imgR)
 
# show output
cv2.imshow("post process result", img_1)
	
cv2.imwrite("./save.png", img_1)
# Press Q on keyboard to  exit
while(1):
	if cv2.waitKey(25) & 0xFF == ord('q'):
		break
 
 
# Closes all the frames
cv2.destroyAllWindows()
