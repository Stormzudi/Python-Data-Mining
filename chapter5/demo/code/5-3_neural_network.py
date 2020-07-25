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
data = pd.read_excel(inputfile, index_col = u'���')  # 导入数据

# 数据是类别标签，要将它转化为数据
# 用1来表示“好”、“是”、“高”这3个属性，用0来表示“坏”、“否”、“低”

data[data == u'好'] = 1
data[data == u'是'] = 1
data[data == u'高'] = 1
data[data != 1] = 0
x = data.iloc[:,:3].as_matrix().astype(int)
y = data.iloc[:,3].as_matrix().astype(int)

model = Sequential() #����ģ��
model.add(Dense(input_dim = 3, output_dim = 10))
model.add(Activation('relu')) #��relu������Ϊ��������ܹ�����ṩ׼ȷ��
model.add(Dense(input_dim = 10, output_dim = 1))
model.add(Activation('sigmoid')) #������0-1�������sigmoid������Ϊ�����

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', class_mode = 'binary')
#����ģ�͡��������������Ƕ�Ԫ���࣬��������ָ����ʧ����Ϊbinary_crossentropy���Լ�ģʽΪbinary
#���ⳣ������ʧ��������mean_squared_error��categorical_crossentropy�ȣ����Ķ������ļ���
#��ⷽ������ָ����adam������sgd��rmsprop�ȿ�ѡ

model.fit(x, y, nb_epoch = 1000, batch_size = 10) #ѵ��ģ�ͣ�ѧϰһǧ��
yp = model.predict_classes(x).reshape(len(y)) #����Ԥ��


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