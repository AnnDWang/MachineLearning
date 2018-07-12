# -*- coding:utf-8 -*-
import numpy as np

# k-means聚类
# 自己按照书上的方法实现的lvq 学习向量量化
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

def lvq(dataset,k,itera_nums):
    return 0