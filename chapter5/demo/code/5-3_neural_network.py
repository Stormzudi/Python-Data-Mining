#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : 5-3_neural_network.py
# @Author: Stormzudi
# @Date  : 2020/7/24 22:27

# 使用神经网络算法预测销量高低

import pandas as pd
from sklearn.metrics import confusion_matrix  # 导入混淆矩阵函数
from keras.models import Sequential
from keras.layers.core import Dense, Activation

# 参数初始化
inputfile = '../data/sales_data.xls'
data = pd.read_excel(inputfile, index_col = u'序号')  # 导入数据

# 数据是类别标签，要将它转化为数据
# 用1来表示“好”、“是”、“高”这3个属性，用0来表示“坏”、“否”、“低”

data[data == u'好'] = 1
data[data == u'是'] = 1
data[data == u'高'] = 1
data[data != 1] = 0
x = data.iloc[:,:3].values.astype(int)
y = data.iloc[:,3].values.astype(int)

model = Sequential()  # 建立模型
model.add(Dense(input_dim = 3, output_dim = 10))
model.add(Activation('relu'))  # 用relu函数作为激活函数，能够答复提供准确度
model.add(Dense(input_dim = 10, output_dim = 1))
model.add(Activation('sigmoid'))  # 由于是0-1输出，用sigmoid函数作为激活函数

model.compile(loss = 'binary_crossentropy', optimizer = 'adam')
# 编译模型。由于我们做的是二元分类，所以我们指定损失函数为binary_crossentropy, 以及模式为binary
# 另外常见的损失函数还有mean_squared_ error、 categorical_crossentropy等， 请阅读帮助文件。
# 求解方法我们指定用adam，还有sgd, rmsprop等 可选。

model.fit(x, y, batch_size=10, nb_epoch=1000)  # 训练模型，学习100次
yp = model.predict_classes(x).reshape(len(y))   # 分类预测
print(yp)


# 画出混淆矩阵
def cm_plot(y, yp):
    """
    输入：y为真实值，yp为预测值
    """
    cm = confusion_matrix(y, yp)  # 混淆矩阵
    import matplotlib.pyplot as plt  # 导入作图库
    plt.matshow(cm, cmap=plt.cm.Greens)  # 画混淆矩阵图，配色风格使用cm.Greens，更多风格请参考官网。
    plt.colorbar()  # 颜色标签
    for x in range(len(cm)):  # 数据标签
        for y in range(len(cm)):
            plt.annotate(cm[x, y], xy=(x, y), horizontalalignment='center', verticalalignment='center')
    plt.ylabel('True label')  # 坐标轴标签
    plt.xlabel('Predicted label')  # 坐标轴标签
    plt.show()  # 显示作图结果

cm_plot(y, yp)