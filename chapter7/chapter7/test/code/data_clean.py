# -*- coding: utf-8 -*-


"""
# 数据清洗，过滤掉不符合规则的数据
"""
import pandas as pd


def clean_data():
    """
    删除
    """
    data = {
        'a': [1, 2, 3, 4, 0, 6, 7, 8, 9, 10],
        'b': [2015, 0, 2017, 2001, 2002, 2004, 2006, 2008, 2010, 2012],
        'c': [1.5, 1.7, 3.6, 2.4, 2.9, 1.5, 1.7, 3.6, 2.4, 2.9]
    }
    frame1 = pd.DataFrame(data, columns=['a', 'b', 'c'])
    index1 = frame1['a'] != 0
    index2 = frame1['b'] != 0
    frame1 = frame1[index1 & index2]  # 删除掉‘a’和‘b’列下的为0的值
    print(frame1)


def clean():
    """
    功能：清晰数据，将data中的空值，或者是0，删除这行样本
    """
    datafile = '../data/air_data.csv'  # 航空原始数据,第一行为属性标签
    cleanedfile = '../tmp/data_cleaned.xls'  # 数据清洗后保存的文件
    data = pd.read_csv(datafile, encoding='utf-8')   # 读取原始数据，指定UTF-8编码（需要用文本编辑器将数据装换为UTF-8编码）
    data = data[data['SUM_YR_1'].notnull()*data['SUM_YR_2'].notnull()]  # 票价非空值才保留

    # 只保留票价非零的，或者平均折扣率与总飞行公里数同时为0的记录。
    index1 = data['SUM_YR_1'] != 0
    index2 = data['SUM_YR_2'] != 0
    index3 = (data['SEG_KM_SUM'] == 0) & (data['avg_discount'] == 0)  # 该规则是“与”
    data = data[index1 | index2 | index3]  # 该规则是“或”

    data.to_excel(cleanedfile)  # 导出结果


def clean_five_index():
    """
    功能：清洗air_data_1.xlsx中的数据，
         将指标中'FLIGHT_COUNT'、'SEG_KM_SUM'、'LAST_TO_END'、'AVG_DISCOUNT'中为0的样本
    输出：
    """
    datafile = '../data/air_data_1.xlsx'  # 航空原始数据,第一行为属性标签
    cleaned_file = '../tmp/data_cleaned_five.xls'  # 数据清洗后保存的文件
    data = pd.read_excel(datafile)  # 读取原始数据，指定UTF-8编码
    index1 = data['FLIGHT_COUNT'] != 0
    index2 = data['SEG_KM_SUM'] != 0
    index3 = data['LAST_TO_END'] != 0
    index4 = data['AVG_DISCOUNT'] != 0

    data = data[index1 & index2 & index3 & index4]  # 该规则是“和”
    data.to_excel(cleaned_file)  # 导出结果


def del_wrong_data():
    """
        功能：删除data_cleaned_five.xlsx中的异常数据，
        输出：cleaned_file处理后的文件
        """
    datafile = '../data/data_cleaned_five.xls'  # 航空原始数据,第一行为属性标签
    cleaned_file = '../tmp/data_del_wrong_data.xls'  # 数据清洗后保存的文件
    data = pd.read_excel(datafile, sheet_name= 'Sheet2')

    # 将每列进行异常值的删除
    for i in data.columns:
        data1 = data[i]
        # 设定法则的左右边界
        left = data1.mean() - 6 * data1.std()
        right = data1.mean() + 6 * data1.std()
        # 获取在范围内的数据
        index = (left < data1) & (data1 < right)
        data = data[index]
    data.to_excel(cleaned_file)  # 导出结果


if __name__ == '__main__':
    # clean_data()
    # clean()
    # clean_five_index()
    del_wrong_data()









