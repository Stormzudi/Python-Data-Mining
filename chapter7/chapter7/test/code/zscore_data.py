# -*- coding: utf-8 -*-


"""
# 标准差标准化处理
"""

import pandas as pd
import numpy as np


def Standardized(data):
    """
    函数功能：标准化处理，不调用库函数
    输入：待标准化的数据
    输出：标准化后的数据
    """
    data = np.array(data)
    row = np.shape(data)[0]
    col = np.shape(data)[1]
    mean = np.mean(data, axis=0)  # 获取每个指标的平均值
    arr_std = np.std(data, axis=0, ddof=1)  # 获取每个指标的标准差
    data_st = np.zeros((row, col))  # 标准化后的数据
    for i in range(col):
        st = (data[:, i] - mean[i]) / arr_std[i]
        data_st[:, i] = st
    return data_st


if __name__ == '__main__':
    datafile = '../data/data_del_wrong_data.xls'  # 需要进行标准化的数据文件；
    zscoredfile = '../tmp/zscoreddata.xls'  # 标准差化后的数据存储路径文件；

    # 标准化处理
    data = pd.read_excel(datafile)
    data_st = Standardized(data)

    data_st = pd.DataFrame(data_st, columns=['L', 'R', 'F', 'M', 'C'])  # 将ndarray类型转化成dataFarme类型
    data_st.to_excel(zscoredfile, index=False)  # 数据写入
