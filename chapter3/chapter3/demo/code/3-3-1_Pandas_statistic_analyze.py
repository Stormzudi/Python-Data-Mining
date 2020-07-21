#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : 3-3-1_Pandas_statistic_analyze.py
# @Author: Stormzudi
# @Date  : 2020/7/20 16:49


"""
3.3.1基本统计特征函数
"""
import pandas as pd
import numpy as np

# 运用字典生成DataFrame数据集
data = {
    'state': ['Ohio','Ohio','Ohio','Nevada','Nevada'],
    'year': [2015,2016,2017,2001,2002],
    'pop': [1.5,1.7,3.6,2.4,2.9]
}
frame2 = pd.DataFrame(data, index=['one','two','three','four','five'],columns=['year','state','pop'])

# 对数据样本的指标"pop"进行基本统计特征分析
print(frame2['pop'].sum())  # 计算数据样本的总和(按列计算)。
print(frame2['pop'].mean())  # 计算数据样本的算术平均数。
print(frame2['pop'].var())  # 计算数据样本的方差。
print(frame2['pop'].std())  # 计算数据样本的标准差。


# 计算数据样本的Spearman(Pearson)相关系数矩阵
D = pd.DataFrame([range(1, 8), range(2, 9)])  # 生成样本D，一行为1~7，一行为2~8
D.corr(method='pearson')  # 计算相关系 数矩阵
S1 = D.loc[0]  # 提取第一行
S2 = D.loc[1]  # 提取第二行
var = S1.corr(S2, method='pearson')  # 计算S1. s2的相关系数
print(var)


# 计算数据样本的协方差矩阵
D = pd.DataFrame(np.random.randn(6, 5))  # 产生6*5随机矩阵
D.cov()  # 计算协方差矩阵
print(D.cov())


# 计算数据样本的偏度(三阶矩) /峰度(四阶矩)。
D = pd.DataFrame(np.random.randn(6,5)) #产生6X5随机矩阵
D.skew()
D.kurt()


# 直接给出样本数据的描述
D = pd.DataFrame(np.random.randn(6, 5))
D.describe()


