#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : 3-2-2_statistics_analyze.py
# @Author: Stormzudi
# @Date  : 2020/7/19 22:06


"""
绘制菜品A. B、C在某段时间的销售量的分布图
"""
import matplotlib.pyplot as plt
import seaborn as sns   # seaborn画出的图更好看，且代码更简单，缺点是可塑性差
sns.set(color_codes=True)  # seaborn设置背景

# 菜品A、B、C的数量为
num = [330, 110, 560]

# 绘制饼图
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
labels = ['菜品A','菜品B','菜品C']
plt.pie(num, labels=labels, autopct='%1.1f%%', shadow=False, startangle=150)
plt.title("菜品销售量分布(饼图)")
plt.axis('equal')   # 该行代码使饼图长宽相等


# 绘制直方图
fig, ax = plt.subplots()
ax.bar(labels,num, color='SkyBlue', label='Men')
ax.set_title('菜品的销售量分布(条形图)')
plt.show()
