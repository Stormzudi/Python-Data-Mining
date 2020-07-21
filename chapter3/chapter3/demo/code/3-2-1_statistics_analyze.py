#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : 3-2-1_statistics_analyze.py
# @Author: Stormzudi
# @Date  : 2020/7/19 21:21


"""
表3-2是描述菜品“捞起生鱼片”在2014年第二个季度的销售数据，
通过表中数据绘制销售量的频率分布表、频率分布图，对该定量数据做出相应的分析。
"""

import pandas as pd
import matplotlib.pyplot as plt


catering_sale = '../data/catering_sale.xls'  # 餐饮数据
data = pd.read_excel(catering_sale, usecols=[1])
data = data.fillna(0).values  # 将缺失值替换为那个常数值0

number = len(data)  # 统计出分组
# 设置长度为 8 的分组
num0 = []; num1 = []; num2 = []
num3 = []; num4 = []; num5 = []
num6 = []; num7 = []

for i in range(number):
    if data[i,0]<500:
        num0.append(data[i,0])
    elif data[i,0]<1000:
        num1.append(data[i,0])
    elif data[i,0]<1500:
        num2.append(data[i,0])
    elif data[i,0]<2000:
        num3.append(data[i,0])
    elif data[i,0]<2500:
        num4.append(data[i,0])
    elif data[i,0]<3000:
        num5.append(data[i,0])
    elif data[i,0]<3500:
        num6.append(data[i,0])
    elif data[i,0]<4000:
        num7.append(data[i,0])

# 统计频数
num = [len(num0), len(num1), len(num2), len(num3), len(num4), len(num5), len(num6), len(num7)]
print(num)

# 绘制直方图
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.bar(['0~500','500~1000','1000~1500','1500~2000','2000~2500','2500~3000','3000~3500','>3500'] ,
        num, label='季度销售额频率分布直方图')
plt.legend()
plt.xlabel('日销售额/元')
plt.ylabel('频数')
plt.title('销售额的频率分布直方图')
plt.show()

