#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : 3-2-4_statistics_analyze.py
# @Author: Stormzudi
# @Date  : 2020/7/20 15:19

"""
功能：周期性分析
"""

import pandas as pd
import matplotlib.pyplot as plt

catering_sale = '../data/catering_fish_congee.xls'  # 餐饮数据
data = pd.read_excel(catering_sale, usecols=[1])
data = data.fillna(0).values  # 将缺失值替换为那个常数值0

# 绘制折线图
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.plot(data, 'ko--')
plt.ylabel('日销售量')
plt.title('销售额的折线图')
plt.show()
