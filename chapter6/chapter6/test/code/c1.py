#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : c1.py
# @Author: ZhuNian
# @Date  : 2020/6/26 15:07

"""
1. 实现bool值的运算
2. pd.reindex()
"""

import pandas as pd  # 导入数据分析库Pandas
from scipy.interpolate import lagrange  # 导入拉格朗日插值函数


inputfile = '../data/missing_data.xls'  # 输入数据路径,需要使用Excel格式；
outputfile = '../tmp_znz/missing_data_processed.xls'  # 输出数据路径，使用Excel格式
data = pd.read_excel(inputfile, header=None)  # 读入数据

n = 7
k = 5

y = data[2][list(range(n-k, n))+list(range(n+1, n+1+k))]  # 获取前后的五个数(报错)
y2 = data[2].reindex(list(range(n-k, n)) + list(range(n+1, n+1+k)))  # 获取前后的五个数
# print(y)
print(y2)


# 将空闲的值删除掉，bool值类型的运算
y = y[y.notnull()]


print(y)
print(y.index)
print(list(y))
print(lagrange(y.index, list(y)))  # 返回拉格朗日插值函数（多项式函数）
print(lagrange(y.index, list(y))(1))  # 将1带入到函数中

