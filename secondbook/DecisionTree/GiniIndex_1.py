# -*- coding:utf-8 -*-
import numpy as np

# 基尼指数来选取划分属性
# 西瓜书 79页
# 目前的版本，还有一些bug

# 获取数据的方法
def readFile(filename,split_str):
    # files = open(filename)
    # 如果读取不成功试一下
    files = open(filename, "r", encoding="utf8")
    data = []
    for line in files.readlines():
        item = line.strip().split(split_str)
        data.append(item)
    return data

dataset=readFile("xigua3.txt",',')

# 计算数据集的纯度基尼值
def getGini(dataset):
    if len(dataset)!=0:
        labels = {}
        for data in dataset:
            if data[-1] not in labels:
                labels[data[-1]] = 1
            else:
                labels[data[-1]] = labels[data[-1]] + 1
        # 计算基尼值
        total = 1
        num = len(dataset)
        for i in labels:
            p=labels[i]/num
            total=total-p*p
        return total

# 获取某个属性下的取值
# i 表示第几列数据
def getFeatureValues(dataset, i):
    temp_features_dict = {}
    for data in dataset:
        if data[i] in temp_features_dict:
            temp_features_dict[data[i]].append(data)
        else:
            temp_features_dict[data[i]] = [data]
    return temp_features_dict

# 计算不同属性的GiniIndex
def getGiniIndex(dataset):
    m, n = np.shape(dataset)
    feature_gain_dict = []
    # m 表示了特征数目, i表示当前是第几个特征
    for i in range(1, 7):
        # 第i个特征，第一列是序号，排除在外
        temp_features_dict = getFeatureValues(dataset, i)

        temp_features_gini_num_dict = {}
        for j in temp_features_dict:
            temp_features_gini_num_dict[j] = [len(temp_features_dict[j]),
                                                 getGini(temp_features_dict[j])]
        total_gini=0.0
        for k in temp_features_gini_num_dict:
            total_gini +=  temp_features_gini_num_dict[k][0]/m * temp_features_gini_num_dict[k][1]
        feature_gain_dict.append([total_gini, i])
        feature_gain_dict.sort()

    return feature_gain_dict

def gini_func(dataset):
    # 根据西瓜书第74页决定递归结束条件，
    # （1）当前结点包含的样本全属于同一类别
    # （2）当前属性集为空，或是所有样本在所有属性上取值相同
    # （3）当前结点包含的样本集合为空
    if len(dataset) == 0:
        # （3）当前结点包含的样本为空，不能划分
        print('当前数据集为空，递归结束')
        return '递归结束'
    else:
        # 当前结点内部有样本
        # 首先判断当前结点包含的样本是否都属于同一类别
        cur_dataset_labels_dict = {}
        for data in dataset:
            if data[-1] not in cur_dataset_labels_dict:
                cur_dataset_labels_dict[data[-1]] = 1
        if len(cur_dataset_labels_dict) == 1:
            print('当前为叶子节点，递归结束')
            for data1 in dataset:
                print(data1)
            return '递归结束'
        # 判断样本中所有属性取值是否相同
        for i in range(1, 7):
            temp_features_dict = getFeatureValues(dataset, i)
            if len(temp_features_dict) > 1:
                # 该属性元素取值不同，计算信息增益
                cur_feature_dict = getGiniIndex(dataset)
                # 获取第一个feature作为划分属性
                cur_split_feature = cur_feature_dict[0][1]
                print('当前划分属性为第' + str(cur_split_feature) + '个属性')
                # 用该属性进行划分
                cur_split_feature_dict = getFeatureValues(dataset, cur_split_feature)
                for j in cur_split_feature_dict:
                    cur_split_feature_dataset = cur_split_feature_dict[j]
                    # 递归获取下一个划分属性
                    gini_func(cur_split_feature_dataset)
                # 不继续循环
                break;
        print('递归结束')
        return '递归结束'


gini_func(dataset)
