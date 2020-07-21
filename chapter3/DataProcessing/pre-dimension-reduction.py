#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : pre-dimension-reduction.py
# @Author: ZhuNian
# @Date  : 2020/6/25 22:46


"""
1 多个文件进行特征值匹配，（按照文件中的共同特征值进行合并）
2 将数据进行降维处理
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

# 读取四张表的数据


prior = pd.read_csv("./instacart-market-basket-analysis/instacart/order_products__prior.csv")
products = pd.read_csv("./instacart-market-basket-analysis/instacart/products.csv")
orders = pd.read_csv("./instacart-market-basket-analysis/instacart/orders.csv")
aisles = pd.read_csv("./instacart-market-basket-analysis/instacart/aisles.csv")

# 合并到一张表中
_mg = pd.merge(prior, products, on=['product_id', 'product_id'])
_mg = pd.merge(_mg, orders, on=['order_id', 'order_id'])
_mg = pd.merge(_mg, aisles, on=['aisle_id', 'aisle_id'])

# print(_mg.head())
# print(_mg.columns)

# 实现交叉表
cross = pd.crosstab(_mg['user_id'], _mg['aisle'])
print(cross.head())

# 主成分分析PCA
pca = PCA(n_components=0.9)  # 保留90%的信息量
data = pca.fit_transform(cross)
print(np.shape(data))


