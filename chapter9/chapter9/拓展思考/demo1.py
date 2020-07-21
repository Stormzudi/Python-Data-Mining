#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : demo1.py
# @Author: ZhuNian
# @Date  : 2020/7/13 17:56


"""
功能：运用决策树来实现空气指标的分类
题目：第九章拓展部分

"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder  # 用于序列化
from sklearn.tree import DecisionTreeClassifier  # 导入决策树分类器
from sklearn.model_selection import cross_val_score  # 导入计算交叉检验值的函数cross_val_score


fr = open('lenses.txt', 'r')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]  # 去掉空格，提取每组数据的类别，保存在列表里
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']  # 特征标签
lenses_target = [row[-1] for row in lenses]  # 目标标签
# print(lenses_target)

lenses_list = []  # 保存lenses数据的临时列表
lenses_dict = {}  # 保存lenses数据的字典，用于生成pandas
for each_label in lensesLabels:
    # 提取信息，生成字典
    for each in lenses:
        lenses_list.append(each[lensesLabels.index(each_label)])
    lenses_dict[each_label] = lenses_list
    lenses_list = []

lenses_data = pd.DataFrame(lenses_dict)  # 生成pandas.DataFrame
# print(lenses_data)
le = LabelEncoder()  # 创建LabelEncoder()对象，用于序列化
for col in lenses_data.columns:
    # 为每一列序列化
    lenses_data[col] = le.fit_transform(lenses_data[col])

lenses_target = le.fit_transform(lenses_target) # 将标签序列化
# print(lenses_target)
# print(lenses_data)

lenses_data = np.array(lenses_data)
lenses_target = np.array(lenses_target)

# 创建一颗基于基尼系数的决策树
clf = DecisionTreeClassifier(criterion="gini", max_features=None)
score = cross_val_score(clf, lenses_data, lenses_target, cv = 10) # 使用10折交叉验证

clf.fit(lenses_data, lenses_target) # fit()训练模型
# 待预测的测试集
test = [
    [1, 1, 1, 0],
    [2, 0, 1, 0],
    [0, 1, 0, 1]]
pre_text_tree = clf.predict(test) # 决策树，函数预测
print('运用 决策树 预测分类结果：',pre_text_tree)