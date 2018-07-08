# -*- coding:utf-8 -*-
import numpy as np

# BP神经网络的实现
# 参考西瓜书101-106页


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


dataset=np.loadtxt('xigua.txt',delimiter=',')

# 带标签的数据
data=dataset[:,1:10]
# 取特征
features_data=data[:,0:8]
# 数目
m,n=np.shape(data) # 行数、列数
# 判断输入的数目
input_num=n-1
# 判断输出的数目
output_num=len(np.unique(data[:,-1]))



# 隐层数目
# 隐层节点数目必须小于m-1，m为训练样本数，否则网络模型的系统训练误差与训练样本的特性无关而趋于0，即建立的网络模型没有泛化能力，没有任何实用价值
# 训练样本数必须多于网络模型的连接权数，一般为2-10倍，否则，样本必须分成几部分并且采用轮流训练的方法才能得到可靠的神经网络模型
h_num=10

# 学习率
eta=0.1

# 设置默认阈值为1
#features_data=np.c_[features_data,np.ones(m)]

# 输入层与隐藏层之间的权值矩阵
v_input_hidden=np.mat(np.random.random((input_num,h_num)))
# 加入阈值的权重
# 隐藏层阈值矩阵
h_m=np.random.random((1,h_num))

# 隐层与输出层之间的权值矩阵
w_hidden_output=np.mat(np.random.random((h_num,output_num)))

# 输出层阈值矩阵
o_m=np.random.random((1,output_num))

def BP(dataset):
    for data in dataset:
        features=data[0:8]
        y_real=data[-1]
        hidden_values=np.dot(v_input_hidden.transpose(),np.mat(features).transpose())
        # 减去阈值
        hidden_values=hidden_values-h_m.transpose()
        # sigmoid函数取值
        for y in range(h_num):
            hidden_values[y][0]=sigmoid(hidden_values[y][0])
            # if temp>=0.5:
            #     hidden_values[y][0]=1
            # else:
            #     hidden_values[y][0]=0
        # 根据隐藏层计算输出层
        output_values=np.dot(w_hidden_output.transpose(),hidden_values)
        output_values = output_values - o_m.transpose()
        # sigmoid 函数取值
        for y in range(output_num):
            output_values[y][0]=sigmoid(output_values[y][0])
            # # 得到输出矩阵
            # if temp>=0.5:
            #     output_values[y][0]=1
            # else:
            #     output_values[y][0]=0
        g=np.ones(np.shape(output_values))
        for n in range(output_num):
            g[n]=output_values[n] * (1 - output_values[n]) * (y_real - output_values[n])
        g=eta*g
        # 计算隐层和输出层之间权值的变化
        delta_w=np.dot(hidden_values,np.mat(eta*g).transpose())
        # 计算输出阈值的变化
        delta_o_m=-eta*g
        # 计算eh中间量
        eh=hidden_values*(1-hidden_values).T*(w_hidden_output*g)
        # 计算 输入层和隐层之间的权值变化
        delta_v=(eta*eh*features).T
        # 计算隐层阈值变化
        delta_h_m=-eta*eh
        # 更新各个值
        v_input_hidden=v_input_hidden-delta_v
        w_hidden_output=w_hidden_output-delta_w
        h_m=h_m-delta_h_m
        o_m=o_m-delta_o_m
    return v_input_hidden,w_hidden_output,h_m,o_m

def sigmoid(x):
    y=1.0/(1+np.exp(-x))
    return y

# 初始化所有的权值
BP(data)