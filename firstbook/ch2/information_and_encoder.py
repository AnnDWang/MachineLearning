# -*- coding: utf-8 -*
#信息熵函数
import numpy as np
def entropy(P):
    """根据每个样本出现的概率，计算信息量，输入的P是数据集上每个数值统计的频率向量"""
    return -np.sum(P*np.log2(P))

P=[0.25,0.25,0.25,0.25]
H=entropy(P)
print('小炒的信息量：{}'.format(H))
#数据集中，样本标签的信息量的多少=编码数据集中所有样本标签所需要的最短字符的长度

p_1=[0.5,0.5]
frac_1=50.0/100.0#奇数题的比例

p_2=[0.5,0.5]
frac_2=50.0/100.0#偶数题的比例

H=frac_1*entropy(p_1)+frac_2*entropy(p_2)

print('小炒的信息量：{}'.format(H))