# -*- coding:utf-8 -*-
import numpy as np

# DBSCAN聚类
# 自己按照书上的方法实现的DNSCAN 密度聚类
# 好像有点问题，在计算密度可达的所有样本部分，与书上结果不一致


dataSet = [
        # 1
        [0.697, 0.460],
        # 2
        [0.774, 0.376],
        # 3
        [0.634, 0.264],
        # 4
        [0.608, 0.318],
        # 5
        [0.556, 0.215],
        # 6
        [0.403, 0.237],
        # 7
        [0.481, 0.149],
        # 8
        [0.437, 0.211],
        # 9
        [0.666, 0.091],
        # 10
        [0.243, 0.267],
        # 11
        [0.245, 0.057],
        # 12
        [0.343, 0.099],
        # 13
        [0.639, 0.161],
        # 14
        [0.657, 0.198],
        # 15
        [0.360, 0.370],
        # 16
        [0.593, 0.042],
        # 17
        [0.719, 0.103],
        # 18
        [0.359, 0.188],
        # 19
        [0.339, 0.241],
        # 20
        [0.282, 0.257],
        # 21
        [0.748, 0.232],
        # 22
        [0.714, 0.346],
        # 23
        [0.483, 0.312],
        # 24
        [0.478, 0.437],
        # 25
        [0.525, 0.369],
        # 26
        [0.751, 0.489],
        # 27
        [0.532, 0.472],
        # 28
        [0.473, 0.376],
        # 29
        [0.725, 0.445],
        # 30
        [0.446, 0.459],
    ]

# 获取一个点的密度可达的数据
def getDensityAvaliable(center,dataset,epislon,temp):
    if center not in temp:
        temp.append(center)
    new_temp=[]
    for data in dataset:
        cur_distance=np.sqrt((center[0]-data[0])*(center[0]-data[0])+(center[1]-data[1])*(center[1]-data[1]))
        if cur_distance<=epislon and cur_distance!=0:
            if data not in temp:
                temp.append(data)
                new_temp.append(data)
    # new_temp[] 是当前点的密度直达对象
    # 遍历temp[] 获取密度可达的点
    for data in new_temp:
        cur_temp=getDensityAvaliable(data,dataset,epislon,temp)
        for cur_data in cur_temp:
            if cur_data not in temp:
                temp.append(cur_data)
    return temp

def dbscan(dataset,epislon,minpts):
    # 先获取核心对象
    center_dots=[]
    for data in dataset:
        cur_data=data
        temp_neighbors=[]
        for data1 in dataset:
            if data1!=cur_data:
                cur_distance=np.sqrt((data1[0]-cur_data[0])*(data1[0]-cur_data[0])+(data1[1]-cur_data[1])*(data1[1]-cur_data[1]))
                if cur_distance<=epislon:
                    temp_neighbors.append(data1)
        if len(temp_neighbors)>=minpts:
            center_dots.append(data)
    # 获取核心对象后，再从核心对象组中一一取出
    classes_dic={}
    i=0
    for center in center_dots:
        # 获取center点密度可达的所有点
        temp=[]
        cur_avaliable=getDensityAvaliable(center,dataset,epislon,temp)
        classes_dic[i]=cur_avaliable
        i+=1
        # 去掉核心对象在当前簇中的点
        for ca in cur_avaliable:
            if ca in center_dots:
                center_dots.remove(ca)

    return classes_dic



final_dict=dbscan(dataSet,0.11,5)

print('最终结果为：'+str(final_dict))

