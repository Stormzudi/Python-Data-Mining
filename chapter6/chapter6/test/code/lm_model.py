# -*- coding: utf-8 -*-
# 构建并测试LM神经网络模型


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  # 交叉验证
from keras.models import Sequential  # 导入神经网络初始化函数
from keras.layers.core import Dense, Activation  # 导入神经网络层函数、激活函数
from sklearn.metrics import confusion_matrix  # 导入混淆矩阵函数
import matplotlib.pyplot as plt  # 导入作图库
from sklearn.metrics import roc_curve  # 导入ROC曲线函数


# 交叉验证
def split(data):
    data = np.array(data)
    X = data[:, 0:3]  # X用于保存特征数据
    y = data[:, -1]   # y用于保存标签数据
    train_set_X, test_set_X, train_set_y, test_set_y = train_test_split(X, y, test_size=0.2, random_state=0)
    return train_set_X, test_set_X, train_set_y, test_set_y


# 建立LM神经网络模型
def LM(data):
    train_set_X, test_set_X, train_set_y, test_set_y = split(data)
    netfile = '../tmp/net.model'  # 构建的神经网络模型存储路径

    net = Sequential()  # 建立神经网络
    net.add(Dense(input_dim=3, output_dim=10))  # 添加输入层（3节点）到隐藏层（10节点）的连接
    net.add(Activation('relu'))  # 隐藏层使用relu激活函数
    net.add(Dense(input_dim=10, output_dim=1))  # 添加隐藏层（10节点）到输出层（1节点）的连接
    net.add(Activation('sigmoid'))  # 输出层使用sigmoid激活函数
    net.compile(loss = 'binary_crossentropy', optimizer = 'adam')  # 编译模型，使用adam方法求解
    net.fit(train_set_X, train_set_y, nb_epoch=100, batch_size=1)  # 训练模型，循环200次
    # net.save_weighpassts(netfile)  # 保存模型
    pre_text_LM = net.predict(test_set_X)  # 函数预测
    pre_text_LM = np.array(pre_text_LM).reshape(-1)

    print('运用"LM"预测结果：', np.around(pre_text_LM, decimals=1))
    print('原本的数据的结果：', test_set_y)

    # 保留一位有效数字
    for i in range(len(pre_text_LM)):
        if pre_text_LM[i] >= 0.1:
            pre_text_LM[i] = 1
        else:
            pre_text_LM[i] = 0
    return test_set_y, pre_text_LM


# 画出混淆矩阵
def cm_plot(data):
    """
    输入：y为真实值，yp为预测值
    """
    y, yp = LM(data)  # 获取真实值和预测值
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


# ROC图像判断模型的准确度
def curve_roc(data):
    test_set_y, pre_text_LM = LM(data)
    pre_text_LM = pre_text_LM.reshape(len(pre_text_LM))  # 预测结果变形

    fpr, tpr, thresholds = roc_curve(test_set_y, pre_text_LM, pos_label=1)
    plt.plot(fpr, tpr, linewidth=2, label = 'ROC of LM', color = 'green')  # 出ROC曲线
    plt.xlabel('False Positive Rate')  # 坐标轴标签
    plt.ylabel('True Positive Rate')  # 坐标轴标签
    plt.ylim(0,1.05)  # 边界范围
    plt.xlim(0,1.05)  # 边界范围
    plt.legend(loc=4)  # 图例
    plt.show()  # 显示作图结果


if __name__ == '__main__':
    datafile = '../data/model.xls'
    data = pd.read_excel(datafile)
    # split(data)
    # LM(data)
    # cm_plot(data)
    curve_roc(data)

