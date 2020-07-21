#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : 01feature_extraction.py
# @Author: ZhuNian
# @Date  : 2020/6/20 17:18


"""
位置：思维导图“机器学习”
章节：01-2  # 获取特征并且数值化
章节：01-3  # 字典数据抽取
章节：01-4  # 中文 特征化
章节：01-5  # tf_df分析问题
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
import jieba   # 用于分段
from sklearn.feature_extraction.text import TfidfVectorizer


def CountVec():
    # 实例化类CountVectorizer
    vector = CountVectorizer()
    # 调用fit_transform输出并转换
    res = vector.fit_transform(["life is short,i like python", "life is too long, i don't like python",
                                "life is too long, i like matlab"])
    # 打印结果
    print(vector.get_feature_names())  # 获取输入字段中，所有的特征
    print(res.toarray())  # 输出具有或者不具有【0，1】此字段


def dictver():
    """
    字典数据抽取
    return: None
    """
    simple = [{'city': '北京', 'temperature': 100},
              {'city': '上海', 'temperature': 200},
              {'city': '广州', 'temperature': 300}]
    # 实例化一个对象
    dict = DictVectorizer()
    dict1 = DictVectorizer(sparse = False)

    # 调用fit_transfrom()
    data = dict.fit_transform(simple)  # 返回的时一个sparse矩阵（三元表）
    data1 = dict1.fit_transform(simple)  # 返回的是一个ndaary矩阵

    # 看出返回特征值的实例
    print(dict.get_feature_names())
    print(data)
    print(data1)


def funcname():
    # 文本
    data = ["今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
            "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
            "如果只用- -种方式了解某样事物，你就不会真正了解它。解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"
            ]
    con = [list(jieba.cut(i)) for i in data]  # 分割一整段，转化成字符串
    c = [' '.join(i) for i in con]  # 列表中的字符串，转化成”有空格的“列表：['今天 很 残酷 ， 明天 更 残酷']...
    print(c)
    return c


def hanzivec():
    """
    中文 特征化
    return: None
    """
    input = funcname()
    cv = CountVectorizer()
    data = cv.fit_transform(input)

    print(cv.get_feature_names())
    print(data.toarray())


def tfdf():
    """
    tf_df分析问题
    return: None
    """
    input = funcname()
    tfidf = TfidfVectorizer()
    data = tfidf.fit_transform(input)
    print(tfidf.get_feature_names())
    print(data.toarray())

if __name__ == '__main__':
    # CountVec()
    # dictver()
    # funcname()
    hanzivec()
    # tfdf()