# -*- coding: utf-8 -*
#1.4小节 逻辑分类2：线性分类模型
import numpy as np

#函数y=1*x^2+0*x+0
y=np.poly1d([1,0,0])
y(-7)

#d_yx 导函数
d_yx=np.polyder(y)
d_yx(-7)

import random

#随机选择一个起点
x_0=random.uniform(-10,10)
y_0=random.uniform(-10,10)
#uniform() 方法将随机生成下一个实数，它在 [x, y) 范围内。
print(x_0,y_0)

def step(x,d_yx):
    alpha=0.2
    return x-alpha*d_yx(x)

step(x_0,d_yx)

x=x_0
x_list=[x]
for i in range(10):
    x=step(x,d_yx)
    x_list.append(x)

print(x_list)

def score(x, w, b):
    return np.dot(x, w)+b

def softmax(s):
    return np.exp(s) / np.sum(np.exp(s),axis=0)

#损失函数对w的偏导数
def d_loss_func(X,w,b,y,w_i):
    s=score(X,w,b)
    y_p=softmax(s)
    return np.mean(w_i*(y_p-y))

#损失函数对b的偏导数
def d_b_loss(X,w,b,y,d_obj=1):
    s=score(X,w,b)
    y_p=softmax(s)
    return np.mean(d_obj*(y_p-y))

#aplha是一个可调节的模型参数
def step(X,w,b,y,d_obj,loss_func):
    alpha=0.2
    return w-alpha*loss_func.__call__(X,w,b,y,d_obj)
#这里的return本身应该是 return w_i-alpha*loss_func.__call__(X,w,b,y,d_obj)，但是不知道w_i是哪里来的

class GDOptimizer:
    """梯度下降优化器"""
    def optimize(X,y):
        w1=random.uniform(0,1)#随机选择一个起点
        w2=random.uniform(0,1)
        b=random.uniform(0,1)
        w=[w1,w2]
        for i in range(100):
            w1=step(X,w,b,y,w1,d_loss_func)
            w2=step(X,w,b,y,w2,d_loss_func)
            b=step(X,w,b,y,b,d_b_loss)
            w=[w1,w2]

#这套流程就是梯度下降

#均值
x=np.array([1,2,3,4,5])
assert np.mean(x)==np.sum(x) / 5

#方差
assert np.std(x) ==np.sqrt(np.mean((x-np.mean(x))**2))
# **两个乘号就是乘方，比如2**4,结果就是2的4次方，结果是16

#两个特征向量
f1=np.array([0.2,0.5,1.1]).reshape(-1,1)
f2=np.array([-100.0,56.0,-77.0]).reshape(-1,1)

#计算归一化
f1_scaled=(f1-np.mean(f1))/np.std(f1)
f2_scaled=(f2-np.mean(f2))/np.std(f2)

#使用scikit-learn封装的函数计算归一化
import sklearn.preprocessing as preprocessing

scaler=preprocessing.StandardScaler()
f1_sk_scaled=scaler.fit_transform(f1)
f2_sk_scaled=scaler.fit_transform(f2)

assert np.allclose(f1_sk_scaled,f1_scaled) and np.allclose(f2_sk_scaled,f2_scaled)
#使用np.allclose()检测两个矩阵是否相同
