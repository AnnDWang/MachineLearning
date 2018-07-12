# -*- coding:utf-8 -*-
import numpy as np

# k-means聚类
# 自己按照书上的方法实现的k均值聚类
# 但是停止条件没有去判断均值向量的更新，而是选择了一定的迭代次数

dataSet = [
        # 1
        [0.697, 0.460,1],
        # 2
        [0.774, 0.376,1],
        # 3
        [0.634, 0.264,1],
        # 4
        [0.608, 0.318,1],
        # 5
        [0.556, 0.215,1],
        # 6
        [0.403, 0.237,1],
        # 7
        [0.481, 0.149,1],
        # 8
        [0.437, 0.211,1],
        # 9
        [0.666, 0.091,0],
        # 10
        [0.243, 0.267,0],
        # 11
        [0.245, 0.057,0],
        # 12
        [0.343, 0.099,0],
        # 13
        [0.639, 0.161,0],
        # 14
        [0.657, 0.198,0],
        # 15
        [0.360, 0.370,0],
        # 16
        [0.593, 0.042,0],
        # 17
        [0.719, 0.103,0],
        # 18
        [0.359, 0.188,0],
        # 19
        [0.339, 0.241,0],
        # 20
        [0.282, 0.257,0],
        # 21
        [0.748, 0.232,0],
        # 22
        [0.714, 0.346,1],
        # 23
        [0.483, 0.312,1],
        # 24
        [0.478, 0.437,1],
        # 25
        [0.525, 0.369,1],
        # 26
        [0.751, 0.489,1],
        # 27
        [0.532, 0.472,1],
        # 28
        [0.473, 0.376,1],
        # 29
        [0.725, 0.445,1],
        # 30
        [0.446, 0.459,1],
    ]

# 特征值列表
labels = ['密度', '含糖率']

def kmeans(dataset,k,itera_nums):
    # k为聚类的数目
    m,n=np.shape(dataset)
    # n为样本特征数目

    # 创建k个随机向量作为初始中心点
    mean_dots={}
    for i in range(0,3):
        temp=np.random.random((1,n))
        mean_dots[i]=temp[0]
    # 确定聚类中心以后，判断每个元素所属的簇
    for itera_num in range(0,itera_nums):
        print('当前迭代次数为：'+str(itera_num))
        # 创建k个簇
        classes_dic={}
        for k in mean_dots:
            classes_dic[k]=[]
        for data in dataset:
            # 计算每个值距离初始中心点的距离
            temp_distance_dict=[]
            for j in mean_dots:
                dots=mean_dots[j]
                distance=np.sqrt((data[0]-dots[0])*(data[0]-dots[0])+(data[1]-dots[1])*(data[1]-dots[1]))
                temp_distance_dict.append((distance,j))
            temp_distance_dict.sort()
            # 所属的分类为
            label=temp_distance_dict[0][1]
            classes_dic[label].append(data)
        for k in classes_dic:
            temp_classes=classes_dic[k]
            new_means=[0.0,0.0]
            # 新的均值中心点
            if len(temp_classes)!=0:
                new_means=np.mean(temp_classes,axis=0)
            print('第'+str(k)+'个簇当前新的中心点为'+str(new_means))
            # 当前聚类的中心
            cur_dots=mean_dots[k]
            mean_dots[k]=new_means
            print('第'+str(k)+'个簇当前旧的中心点为' + str(cur_dots))


    a=1

kmeans(dataSet,3,10)