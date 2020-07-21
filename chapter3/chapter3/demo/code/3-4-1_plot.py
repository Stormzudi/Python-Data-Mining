#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : 3-4-1_plot.py
# @Author: Stormzudi
# @Date  : 2020/7/20 17:16

"""
3.3.3 统计作图函数
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # 导入作图库

plt.rcParams ['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] =False  # 用来正常显示负号
plt.figure(figsize = (7, 5))  # 创建图像区域，指定比例


# (1) plot
x = np.linspace(0,2*np.pi,50) #x坐标输入
y = np.sin(x) #计算对应x的正弦值
plt.plot(x, y, 'bp--') #控制图形格式为蓝色带星虚线， 显示正弦曲线
plt.show()


# (2) pie
# The slices will be ordered and plotted counter-clockwise.
labels = ['Frogs', 'Hogs', 'Dogs','Logs'] #定义标签
sizes = [15, 30, 45, 10]  # 每一块的比例
colors = ['yellowgreen', 'gold', 'lightskyblue','lightcoral']  # 每一块的颜色
explode = (0, 0.1, 0, 0) #突出显示，这里仅仅突出显示第二块(即'Hogs' )
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='81.1f88', shadow=True, startangle=90)
plt.axis('equal') #显示为圆(避免比例压缩为椭圆)
plt.show()


# (3) hist
x = np.random. randn(1000)  # 1000个服从正态分布的随机数
plt.hist(x, 10)  # 分成10组进行绘制直方图
plt. show()


# (4) boxplot
x = np.random.randn(1000) #1000个服从正态分布的随机数
D = pd.DataFrame([x, x+1]).T #构造两列的DataFrame
D.plot(kind = 'box') #调用Series内置的作 图方法画图，用kind参数指定箱形图box
plt.show()


# (5) plot(logx = True)/plot(logy = True)
x = pd.Series(np.exp(np.arange(20))) # 原始数据
x.plot(label = u'原始数据图', legend = True)
plt.show()
x.plot(logy = True, label = u'对数数据图', legend = True)
plt.show()


# (6) plot(yerr = error)
error = np.random.randn(10)  # 定义误差列
y = pd.Series(np.sin(np.arange(10)))  # 均值数据列
y.plot(yerr = error)  # 绘制误差图
plt.show()



