#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
#import random

LocationHeight = 900
BLACK = 2000
ProportionHeight = 200

# 打开二进制数据
def memory_read(data):
    nparr = np.fromstring(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

# 边缘检测
def getCanny(img):
    # 高斯模糊
    binary = cv2.GaussianBlur(img, (3, 3), 2, 2)
    # 边缘检测
    binary = cv2.Canny(binary, 60, 240, apertureSize=3)
    # 膨胀操作，尽量使边缘闭合
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)
    return binary

# 求出面积最大的轮廓
def findMaxContour(contours):
    max_area = 0.0
    for contour in contours:
        temparea = cv2.contourArea(contour)
        if temparea > max_area:
            max_area = temparea
            max_contour = contour
    return max_contour

# 多边形拟合凸包的四个顶点
def getBoxPoint(contour):
    # 多边形拟合凸包
    hull = cv2.convexHull(contour)
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    # box = []
    # for i in approx:
    #     point = []
    #     point.append(int(i[0][0]))
    #     point.append(int(i[0][1]))
    #     box.append(point)
    # print(approx, len(approx))
    approx = approx.reshape((len(approx), 2))
    return approx

# 适配四边形点集
def adaPoint(box, pro):
    box_pro = box;
    if pro != 1.0:
        box_pro = box/pro
        # for point in box:
        #     point_pro = []
        #     point_pro.append(int(point[0] / pro))
        #     point_pro.append(int(point[1] / pro))
        #     box_pro.append(point_pro)
    box_pro = np.trunc(box_pro)
 #   box_pro = box_pro.astype(np.int32)
    return box_pro

# 四边形顶点排序，[top-left, top-right, bottom-right, bottom-left]
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

# 透视变换
def warpImage(img, box):
    w, h = point_distance(box[0], box[1]),\
            point_distance(box[1], box[2])
    # print("box = {}".format(box))
    # print("w = {} h = {}".format(w, h))
    dst_rect = np.array([[0, 0],
                        [w - 1, 0],
                        [w - 1, h - 1],
                        [0, h - 1]], dtype='float32')
    M = cv2.getPerspectiveTransform(box, dst_rect)
    warped = cv2.warpPerspective(img, M, (w, h))
    # print("warpe.shape {}".format(warped.shape[:2]))
    return warped

# 计算长宽
def point_distance(a,b):
    return int(np.sqrt(np.sum(np.square(a - b))))

# 将图片二值化处理
def getImageThreshold(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 10)
    return th2

# 转化矩阵大小
def resizeImage(image, resizeHeight):
    h, w = image.shape[:2]
    # print("old size ",h,w)
    pro = resizeHeight / h
    size = (int(w * pro), int(resizeHeight))
    # print("new size ",size)
    img = cv2.resize(image, size)
    return img

# 将image矩阵转为二进制数据
def BytesImage(image):
    img_encode = cv2.imencode('.jpg', image)[1]
    data_encode = np.array(img_encode)
    str_encode = data_encode.tostring()
    return str_encode

# 文档切边算法
def file_locations(file):
    pro = 1.0
    # print(type(file))
    image = memory_read(file)
    img = image
    h, w = image.shape[:2]
    if h > LocationHeight:
        pro = LocationHeight/h
        img = resizeImage(image, LocationHeight)
    # 获取边缘
    binary = getCanny(img)
    # 寻找边缘
    _, contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # 找出最大面积的边缘
    max_contour = findMaxContour(contours)
    # 得到四边形的顶点
    box = getBoxPoint(max_contour)
    # 适配点
    box = adaPoint(box, pro)
    # 将顶点排序[top-left, top-right, bottom-right, bottom-left]
    box = order_points(box)
    # 透视变化
    warped = warpImage(image, box)
    # 二值化处理
#    warped_th = getImageThreshold(warped)
    return BytesImage(warped), box

# 获取位置图识别区域点point[(左上), (右下)]
def getRecognitionRect(origin_height, origin_width, origin_point, sorce_height, sorce_width):
    sorce_point = []
    width_scale = sorce_width/origin_width
    height_scale = sorce_height/origin_height
    for i in range(len(origin_point)):
        sorce_point.append([int(origin_point[i][0]*width_scale), int(origin_point[i][1]*height_scale)])
    return sorce_point

# 根据识别点截取需要识别的矩阵数据
def getRecognitionImage(image, point):
    img = image[point[0][1]:point[1][1], point[0][0]:point[1][0]]
    return img

# 获取图像黑点的个数
def getBlackPoint(image):
    # 反相图像
    img = cv2.bitwise_not(image)
    # 摊平矩阵
    img = img.flatten()
    # 归一化处理
    img = img / 255
    num = np.sum(img)
    return num

# 转为同比例大小的矩阵做比较
def transImageSize(origin_image, source_image):
    img1 = resizeImage(origin_image, ProportionHeight)
    img2 = resizeImage(source_image, ProportionHeight)
    return img1, img2

# 比较识别区域是否签字
def file_iffillin(origin_file, origin_point, source_file):
    # 读取图像的高度，宽度
    origin_image = memory_read(origin_file)
    source_image = memory_read(source_file)
    origin_height, origin_width = origin_image.shape[:2]
    source_height, source_width = source_image.shape[:2]
    # 获取目标区域
    source_point = getRecognitionRect(origin_height, origin_width, origin_point, source_height, source_width)
    # 获取截取后的图像
    origin_img = getRecognitionImage(origin_image, origin_point)
    source_img = getRecognitionImage(source_image, source_point)
    # 转为同样比例的矩阵
    origin_img, source_img = transImageSize(origin_img, source_img)
    # 二值化处理
    origin_img = getImageThreshold(origin_img)
    source_img = getImageThreshold(source_img)
    # 获取图像的黑点数
    origin_num = getBlackPoint(origin_img)
    source_num = getBlackPoint(source_img)
    print('origin_num = ',origin_num)
    print('source_num = ', source_num)
    # filename = "origin_{}.jpeg".format(random.randint(0, 10000))
    # filename2 = "source_{}.jpeg".format(random.randint(0, 10000))
    # cv2.imwrite(filename, origin_img)
    # cv2.imwrite(filename2, source_img)
    # 判断目标区域黑点数是否大于未签名图片
    if origin_num + BLACK < source_num:
        return 1, source_point, BytesImage(origin_img), BytesImage(source_img)
    else:
        return 0, source_point, BytesImage(origin_img), BytesImage(source_img)

# path = 'transfer_IMG_0.jpg'
# point = [(496, 2912), (1664, 3103)]
# path2 = 'transfer_IMG_1.jpg'
# f = open(path, 'rb')
# data1 = f.read()
# f.close()
# f = open(path2, 'rb')
# data2 = f.read()
# f.close()
#
# if file_iffillin(data1, point, data2) == 1:
#     print('文档已签名')
# else:
#     print('文档未签名')