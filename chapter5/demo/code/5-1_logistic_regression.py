# -*- coding: utf-8 -*-
# 逻辑回归 自动模型
import pandas as pd
from sklearn.linear_model import LogisticRegression as LR
from stability_selection.randomized_lasso import RandomizedLogisticRegression as RLR


# 参数初始化
filename = '../data/bankloan.xls'
data = pd.read_excel(filename)
x = data.iloc[:, :8].values
y = data.iloc[:, 8].values

rlr = RLR()
rlr.fit(x, y)
rlr.get_support()
print('通过随机逻辑回归模型筛选特征结束。')
print(u'有效特征为:', ','.join(data.columns[rlr.get_support()]))
x = data[data.columns[rlr.get_support()]].as_matrix()

lr = LR()
lr.fit(x, y)
print('逻辑回归模型训练结果')
print('模型的平均正确率', lr.score(x, y))