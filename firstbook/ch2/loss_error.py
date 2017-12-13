# -*- coding: utf-8 -*
#和分类模型意义，训练回归模型的思路是设计损失函数，通过梯度下降法寻找是损失函数尽可能小的最优参数组合w,b，因为目标输出值不再是概率值，所以损失函数也同样改变
#使用预测值与真实值得举例平方和来衡量模型预测值与真实值之间的差异：
import numpy as np

#X是训练样本矩阵，w是权重向量，b是偏置向量，y是真实数值
def loss_func(X,w,b,y):
    y_pred=X*w+b
    return -np.mean((y_pred-y)^2)

#平方的意义在于放大那些预测偏差程度打的错误

#平均绝对误差MAE：
def mean_absolute_error(y,y_pred):
    return np.average(np.abs(y-y_pred),axis=0)

#中位绝对误差:
def median_absolute_error(y,y_pred):
    return np.mean(np.abs(y-y_pred))

#均方差MSE:
def mean_squared_error(y,y_pred):
    return np.average((y-y_pred)**2,axis=0)

#均方根差RMSE：
def root_mean_squared_error(y,y_pred):
    return np.sqrt(np.average((y-y_pred)**2,axis=0))

#r方：
def r2_score(y,y_pred):
    sse=((y-y_pred)**2).sum(axis=0,dtype=np.float64)
    sst=((y-np.average(y,axis=0))**2).sum(axis=0,dtype=np.float64)

    #特殊值处理
    if sse==0.0:
        if sst==0.0:
            return 1.0
        else:
            return 0.0
    return 1- sse/sst
