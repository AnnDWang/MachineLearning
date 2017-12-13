# -*- coding: utf-8 -*
#参数化的模型还有一些其它的好处 更低存储 更快的预测速度 模型对事务的识别方式变成了依赖数学表达式和对应的权重矩阵
#所以对于训练好的模型存储变得异常方便。参数化的模型的存储不再依赖训练集数据，直接保存参数矩阵即可，大大节省控件
#在预测数据样本时只需要将样本数据带入数学表达式，大多数愿意接受训练时间长，应用模型做服务时快速响应
#把每个特征对分类结果的作用加起来，就是线性模型。逻辑分类是一种线性模型，可以表示为y=w*x+b,其中w是训练得到的权重参数，x是样本特征数据，b是偏置。
#逻辑分类也叫做逻辑回归，但是本身是用作分类的
#逻辑分类模型预测一个样本分为三步：
#1. 计算线性函数
#2. 从分数到概率的转换。
#3. 从概率到标签的转换
import numpy as np

def score(x, w, b):
    return np.dot(x, w)+b
#dot函数是np中的矩阵乘法

#分数到概率的变换函数有很多种，常用的是Sigmoid函数和Softmax函数，分别适用于不同的场景
#Sigmoid函数适用于只对一种类别进行分类的场景，首先设置函数阈值，当Sigmoid函数的输出值大于阈值，认为是该类别，否认认为不是。

def sigmoid(s):
    return 1. / (1+np.exp(-s))

#sigmoid函数做了两件事情，将输入的分数范围映射在（0，1）之间，以对数的形式完成到0，1的映射，凸显大的分数作用，使其输出的概率更高，抑制小分数的输出概率

#softmax函数是sigmoid函数的多类别版本，可以将输出值对应到多个类别标签，概率值最高的一项就是模型预测的标签
def softmax(s):
    return np.exp(s) / np.sum(np.exp(s),axis=0)
#sum不传参数的时候，是所有元素的总和
#axis的解释可参考链接：http://blog.csdn.net/rifengxxc/article/details/75008427

#softmax函数做了两件事情，将输入的分数映射到0，1之间，所有分数之和为1.以对数的形式完成映射，凸显其中最大的分数并抑制远低于最大分数的其它数值

import matplotlib.pyplot as plt
#Seaborn 是一个matplotlib之上封装的plot类库
#这里我们只是使用Seaborn的样式定义
import seaborn as sns

x=np.arange(-3.0,6.0,0.1)
#创建等差数列，第一个参数起始点，第二个参数终止点，第三个参数步长
scores=np.vstack([x,np.ones_like(x),0.2*np.ones_like(x)])
#np.vstakc()解释见http://blog.csdn.net/csdn15698845876/article/details/73380803
#np.ones_like()返回一个用1填充的跟输入形状和类型一致的数组
plt.plot(x,softmax(scores).T,linewidth=2)#？？？？

plt.show()#显示图像

#将分数扩大、缩小100倍

scores=np.array([2.0,1.0,0.1])
print(softmax(scores))
print(softmax(scores*100))
print(softmax(scores/100))

#分数扩大100倍之后，概率值大的越大，小的越小。分类器对分类结果更加自信，反之缩小100倍之后，分类器对分类的结果很犹豫。

#one-hot编码
#在统计学中，衡量两个概率分布向量的差异程度，叫做交叉熵，熵是信息的别称。
#交叉熵是衡量两种概率分布相同的概率，数值在0，1之间，交叉熵越高越相似，为1则完全相同。

#y是真实标签，p是预测概率
def cross_entropy(y,p):
    return np.sum(y*np.log(p)+(1-y)*np.log(1-p,axis=1))

#X是训练样本矩阵，w是权重向量，b是偏置向量，y是真实标签矩阵
def loss_func(X,w,b,y):
    s=score(X,w,b)
    y_p=softmax(x)
    return -np.mean(cross_entropy(y,y_p))
