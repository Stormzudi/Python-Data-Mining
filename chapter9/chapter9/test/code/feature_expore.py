#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : feature_expore.py
# @Author: ZhuNian
# @Date  : 2020/7/12 19:08

"""
功能：(1)提取照片中的特征
"""

from sklearn import datasets
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def cat_pic(folder_path):
    """
    输入：文件夹
    输出：
    """
    print("Loading dataset ...")
    datalist = datasets.load_files(folder_path)
    """
    datalist是一个Bunch类，其中重要的数据项有
    data: 原始数据
    filenames: 每个文件的名称
    target: 类别标签（子目录的文件从0开始标记了索引）
    target_names: 类别标签（子目录的具体名称）
    输出总文档数和类别数
    """
    fig1 = datalist.filenames[0]
    f = open(fig1, 'rb')  # 以二进制形式打开文件
    img = Image.open(f)
    plt.imshow(img), plt.title('fig1')
    plt.xticks([]), plt.yticks([])  # 隐藏坐标轴
    plt.show()

    # 按照要求分割图片
    img_np = np.array(Image.open(f))   # 以矩阵的形式返回图片像素值
    rows, cols, dims = img_np.shape   # 图片的大小，以及矩阵上对于的值
    x1 = int(rows/2 - 50); x2 = int(rows/2 + 50)
    y1 = int(cols/2 - 50); y2 = int(cols/2 + 50)
    img_np = img_np[x1:x2, y1:y2]  # 分割后的图像

    plt.imshow(img_np), plt.title('f1')
    plt.xticks([]), plt.yticks([])  # 隐藏坐标轴
    plt.show()
    return img_np


def get_red(img):
    redImg = img[:,:,2]
    return redImg

def get_green(img):
    greenImg = img[:,:,1]
    return greenImg

def get_blue(img):
    blueImg = img[:,:,0]
    return blueImg


def feature_div(path):
    """
    输入：第一个样本图片
    输出：水色图像特征与相应的水色类别的部分数据
    """
    img = cat_pic(path)  # 进行图像分割
    # 分离三通道
    img_blue = get_blue(img)/256
    img_green = get_green(img)/256
    img_red = get_red(img)/256

    # 一阶颜色矩阵
    tp1 = []
    tp1.append(sum(sum(img_red))/10000)
    tp1.append(sum(sum(img_green))/10000)
    tp1.append(sum(sum(img_blue))/10000)

    # 二阶颜色矩阵
    tp1.append(np.sqrt(sum(sum(np.square(img_red - tp1[0])))/10000))
    tp1.append(np.sqrt(sum(sum(np.square(img_green - tp1[1])))/10000))
    tp1.append(np.sqrt(sum(sum(np.square(img_blue - tp1[2])))/10000))

    # 三阶颜色矩阵
    tp1.append(np.power(sum(sum(np.power(img_red - tp1[0], 3)))/10000, 1/3))
    tp1.append(np.power(sum(sum(np.power(img_green - tp1[1], 3)))/10000, 1/3))
    tp1.append(np.power(sum(sum(np.power(img_blue - tp1[2], 3)))/10000, 1/3))

    print(tp1)
    # 输出的tp1为：
    # R通道一阶矩	G通道一阶矩	B通道一阶矩	R通道二阶矩	G通道二阶矩	B通道二阶矩	R通道三阶矩	G通道三阶矩	B通道三阶矩


if __name__ == '__main__':
    inputfile = '../data'  # 数据文件
    feature_div(inputfile)

