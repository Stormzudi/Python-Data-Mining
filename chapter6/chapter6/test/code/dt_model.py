# -*- coding: utf-8 -*-
# 构建并测试CART决策树模型


import numpy as np
import pandas as pd  # 导入数据分析库
from sklearn.model_selection import train_test_split  # 交叉验证
from sklearn.tree import DecisionTreeClassifier  # 导入决策树模型
import matplotlib.pyplot as plt  # 导入作图库
from sklearn.metrics import roc_curve  # 导入ROC曲线函数
from sklearn.metrics import confusion_matrix  # 导入混淆矩阵函数


# 交叉验证
def split(data):
    data = np.array(data)
    X = data[:, 0:3]  # X用于保存特征数据
    y = data[:, -1]   # y用于保存标签数据
    train_set_X, test_set_X, train_set_y, test_set_y = train_test_split(X, y, test_size=0.2, random_state=0)
    return train_set_X, test_set_X, train_set_y, test_set_y


# 建立决策树模型
def decision_tree(data):
    """
    输入：split()返回的数据
    输出：交叉验证后20%的数据的预测值、真实值
    """
    train_set_X, test_set_X, train_set_y, test_set_y = split(data)

    tree = DecisionTreeClassifier(criterion="gini", max_features=None)  # 建立决策树模型
    tree.fit(train_set_X, train_set_y)  # 训练数据
    pre_text_tree = tree.predict(test_set_X)  # 决策树，函数预测
    print('运用"决策树"预测分类结果：', pre_text_tree)
    print('运用"原本的"预测分类结果：', test_set_y)
    return test_set_y, pre_text_tree


# 建立决策树模型
def decision_tree_all(data):
    """
    输入：data所有的数据
    输出：all数据的预测值、真实值
    """
    data = np.array(data)
    tree = DecisionTreeClassifier(criterion="gini", max_features=None)  # 建立决策树模型
    tree.fit(data[:, 0:3], data[:, 3])  # 训练数据
    pre_text_tree = tree.predict(data[:, 0:3])  # 决策树，函数预测
    print('运用"决策树"预测分类结果：', pre_text_tree)
    print('运用"原本的"预测分类结果：', data[:, 3])
    return data[:, -1], pre_text_tree


# 画出混淆矩阵
def cm_plot(data):
    """
    输入：y为真实值，yp为预测值
    """
    y, yp = decision_tree(data)  # 获取真实值和预测值
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
    test_set_y, pre_text_tree = decision_tree_all(data)
    pre_text_tree = pre_text_tree.reshape(len(pre_text_tree))  # 预测结果变形

    fpr, tpr, thresholds = roc_curve(test_set_y, pre_text_tree, pos_label=1)
    plt.plot(fpr, tpr, linewidth=2, label = 'ROC of CART', color = 'green')  # 出ROC曲线
    plt.xlabel('False Positive Rate')  # 坐标轴标签
    plt.ylabel('True Positive Rate')  # 坐标轴标签
    plt.ylim(0,1.05)  # 边界范围
    plt.xlim(0,1.05)  # 边界范围
    plt.legend(loc=4)  # 图例
    plt.show()  # 显示作图结果


if __name__ == '__main__':
    datafile = '../data/model.xls'  # 数据名
    treefile = '../tmp/tree.pkl'  # 模型输出名字
    data = pd.read_excel(datafile)  # 读取数据，数据的前三列是特征，第四列是标签
    # split(data)
    # decision_tree(data)
    cm_plot(data)
    # curve_roc(data)





