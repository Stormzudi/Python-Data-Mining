# -*- coding: utf-8 -*-
# @Time    : 2020/6/3 17:42
# @Author  : Zudy
# @FileName: c7.py

'''
1.异常值剔除: 拉以达法则（3σ 法则）
2.缺失值处理: SimpleImputer()
3.缺失值处理：Pandas
'''


import pandas as pd
import numpy as np
from numpy import NaN as NA
from sklearn.impute import SimpleImputer


def yuchang():
    """
    异常值处理
    拉以达法则（3σ 法则）
    """
    inputfile = 'F:/Python/Python_learning/HBUT/预处理/test1_four.xlsx'
    data = pd.read_excel(inputfile, sheet_name='Sheet2', index_col='日期')
    data1 = data['SS81516']

    # 设定法则的左右边界
    left = data1.mean() - 3 * data1.std()
    right = data1.mean() + 3 * data1.std()

    # 获取在范围内的数据
    new_num = data1[(left < data1) & (data1 < right)]
    t_f = (left < data1) & (data1 < right)
    print(t_f)


def queshi():
    """
    缺失值填补
    调用库函数 SimpleImputer()
    """
    data = np.array([[1, 2, 3], [2, NA, 4], [5, 6, 9]])
    imputer = SimpleImputer(missing_values=NA, strategy = "mean")  # 默认的是每列指标
    r = imputer.fit_transform(data)
    print(r)


def Queshi():
    """
    缺失值填补
    运用 DataFrame()
    """
    data = np.array([[1, 2, 3], [2, NA, 4], [5, 6, 9]])
    data = pd.DataFrame(data)

    data_new = data.fillna(data.mean())  # 默认了运用每列的均值进行填补
    print(data_new)


def replace():
    """
    批量替换
    运用 Pandas中的replace()
    """
    data_ = [[1, 2, 3], [2, '?', 4], [5, 6, 9]]
    df = pd.DataFrame(data_)
    data_new = df.replace('?', NA)
    print(data_new)


if __name__ == '__main__':
    yuchang()
    queshi()
    Queshi()
    replace()







