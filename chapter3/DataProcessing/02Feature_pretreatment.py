#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : 02Feature_pretreatment.py
# @Author: ZhuNian
# @Date  : 2020/6/21 16:42

"""
1.特征预处理-归一化
2.特征预处理-标准化
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


def Normalization():
    """
    归一化处理
    不调用库函数
    """
    data = np.array([[90, 2, 10, 40],[60, 4, 15, 45],[75, 3, 13, 46]])
    row = np.shape(data)[0]
    col = np.shape(data)[1]
    D = np.ones((row, col))

    """
    运用函数式编程：
    input：data,D
    output: NL_data
    """
    f = lambda x, max_data, min_data: (x - min_data) / (max_data - min_data)  # 定义函数f
    for i in range(col):
        D[:, i] = f(data[:, i], max(data[:, i]), min(data[:, i]))

        # 可以指定区间[mx, mi], 不再是区间[0, 1]
        # mx = 2; mi = 1
        # D[:, i] = D[:, i]*(mx - mi) + mi
    print(D)


def mm():
    """
    归一化处理
    运用sklearn库的MinMaxScaler()
    """
    mm = MinMaxScaler()
    data = mm.fit_transform([[90, 2, 10, 40],[60, 4, 15, 45],[75, 3, 13, 46]])
    print(data)


def Standardized():
    """
    标准化处理
    不调用库函数
    """
    data = np.array([[90, 2, 10, 40],[60, 4, 15, 45],[75, 3, 13, 46]])
    row = np.shape(data)[0]
    col = np.shape(data)[1]
    mean = np.mean(data, axis=0)  # 获取每个指标的平均值
    arr_std = np.std(data, axis=0, ddof=1)  # 获取每个指标的标准差
    data_st = np.zeros((row, col))  # 标准化后的数据
    for i in range(col):
        st = (data[:, i] - mean[i])/arr_std[i]
        data_st[:, i] = st
    print(data_st)


def standardized():
    """
    标准化处理
    调用库函数
    """
    sd = StandardScaler()
    data = np.array([[90, 2, 10, 40], [60, 4, 15, 45], [75, 3, 13, 46]])
    data = sd.fit_transform(data)
    print(data)


if __name__ == '__main__':
    Normalization()
    mm()
    Standardized()
    standardized()
