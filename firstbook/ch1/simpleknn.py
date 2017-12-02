# -*- coding: utf-8 -*
#code from book《机器学习之路》
import numpy as np #Numpy可以快速操作结构数组

vec1=np.array([1, 2, 3])
vec2=np.array([4, 5, 6])
#欧氏距离
assert np.linalg.norm(vec1-vec2) == np.sqrt(np.sum(np.square(vec1-vec2)))

