# -*- coding: utf-8 -*-


"""
调用Sklearn库中的KMeans来实现K-Means聚类算法
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans  # 导入K均值聚类算法
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def k_means(inputfile):
    """
    功能：实现聚类算法
    输入：待聚类的样本数据集
    输出：聚类中心，各样本对于的类别
    """
    # 读取数据并进行聚类分析
    data = pd.read_excel(inputfile)  # 读取数据

    # 调用k-means算法，进行聚类分析
    k = 5  # 需要进行的聚类类别数
    kmodel = KMeans(n_clusters=k, n_jobs=4)  # n_jobs是并行数，一般等于CPU数较好
    kmodel.fit(data)  # 训练模型

    var_center = kmodel.cluster_centers_  # 聚类中心
    var_labels = kmodel.labels_  # 各样本对应的分类类别

    # 将样本对应的分类类别添加到输入样本集合中
    cleaned_file = '../tmp/zscoreddata_k_means.xls'  # 数据清洗后保存的文件
    data['label'] = pd.Series(var_labels.T)
    data.to_excel(cleaned_file)  # 导出结果
    return data, var_center, var_labels


def plt_k_means(data, var_center):
    """
    功能：画出k_means的雷达图
    输入：待聚类的样本数、聚类中心
    """
    labels = data.columns  # 标签
    k = 5  # 数据个数
    plot_data = var_center  # 聚类中心
    color = ['b', 'g', 'r', 'c', 'y']  # 指定颜色

    angles = np.linspace(0, 2 * np.pi, k, endpoint=False)
    plot_data = np.concatenate((plot_data, plot_data[:, [0]]), axis=1)  # 闭合
    angles = np.concatenate((angles, [angles[0]]))  # 闭合

    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)  # polar参数！！
    for i in range(len(plot_data)):
        ax.plot(angles, plot_data[i], 'o-', color=color[i], label=u'客户群' + str(i), linewidth=2)  # 画线

    ax.set_rgrids(np.arange(0.01, 3.5, 0.5), np.arange(-1, 2.5, 0.5), fontproperties="SimHei")
    ax.set_thetagrids(angles * 180 / np.pi, labels, fontproperties="SimHei")
    plt.legend(loc=4)
    plt.show()


if __name__ == '__main__':
    inputfile = '../tmp/zscoreddata.xls'  # 待聚类的数据文件
    data, var_center, var_labels = k_means(inputfile)
    # print(var_center)  # 查看聚类中心
    # print(var_labels)  # 查看各样本对应的分类类别

    plt_k_means(data, var_center)

