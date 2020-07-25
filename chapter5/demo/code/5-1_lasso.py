#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : 5-1_lasso.py
# @Author: Stormzudi
# @Date  : 2020/7/24 22:27

"""
实现岭回归分析
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge  # 通过sklearn.linermode1加载岭回归方法
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures  # 通过sklearn.preprocessing加载PolynomialFeatures用于创建多项式特征
from sklearn.model_selection import train_test_split  # 交叉验证

filename = '../data/bankloan.xls'
data_df = pd.read_excel(filename)
data = np.array(data_df)
plt.plot(data[:, 8])  # 展示车流量信息


X = data[:, 0:8]  # X用于保存1-4维数据,即属性
y = data[:, 8]  # y用于保存第5维数据,即车流量
poly = PolynomialFeatures(6)  # 用于创建最高次数6次方的的多项式特征,多次试验后决定采用6次
X = poly.fit_transform(X)  # X为创建的多项式特征

train_set_X, test_set_X, train_set_y, test_set_y = train_test_split(X, y, test_size=0.3, random_state=0)
# 将所有数据划分为训练集和测试集, test_ size表示测试集的比例， random_state是随机数种子

clf = Ridge(alpha=1.0, fit_intercept=True)  # 创建岭回归实例
clf.fit(train_set_X, train_set_y)  # 调用fit函数使用训练集训练回归器
score = clf.score(test_set_X, test_set_y)  # 评价拟合值的好坏(最大值：1)
# print(score)
'''
# 利用测试集计算回归曲线的拟合优度，clf. score返回值为0.7620拟合优度,
# 用于评价拟合好坏,最大为1,无最小值,当对所有输入都输出同一个值时,拟合优度为0。
# '''

# 绘制拟合曲线
start = 200  # 画一段200到300范围内的拟合曲线
end = 300
y_pre = clf.predict(X)  # 是调用predict函数的拟合值
time = np.arange(start, end)

fig = plt.figure()  # 定义一个图片
ax = fig.add_subplot(1, 1, 1)
ax.plot(time, y[start:end], label='real')
ax.plot(time, y_pre[start:end],'r', label='predict')  # 展示真实数据(蓝色)以及拟合的曲线(红色)
plt.legend(loc = 'upper left') # 设置图例的位置
props = {
    'title' : 'Traffic flow forecast',
    'xlabel' : 'Period of time[200-300]',
    'ylabel' : 'Number of traffic'
}
ax.set(**props)
plt.show()
