# -*- coding: utf-8 -*
import pandas as pd #pandas是python的数据格式处理类库

#加载泰坦尼克号生存预测数据集
data_train=pd.read_csv("../data/titanic/train.csv")
data_train.info()

data_train.groupby('Survived').count()

data_train.head(3)

print(data_train.head(3))

#数据缺失处理的方法：
#扔掉缺失数据
#按某个统计量补全
#拿模型预测缺失值

#均值填充缺失数据
def set_missing_ages(p_df):
    p_df.loc[(p_df.Age.isnull(),'Age')]=p_df.Age.dropna().mean()
    return p_df

df=set_missing_ages(data_train)

#归一化数据
import sklearn.preprocessing as preprocessing

scaler=preprocessing.StandardScaler()
df['Age_scaled']=scaler.fit_transform(data_train['Age'])
df['Fare_scaled']=scaler.fit_transform(data_train['Fare'])

#归一化有问题，git没有加载出源码，之后处理

#处理Cabin特征
def set_cabin_type(p_df):
    p_df.loc[(p_df.Cabin.notnull()),'Cabin']="Yes"
    p_df.loc[(p_df.Cabin.isnull()),'Cabin']="No"
    return p_df

df=set_cabin_type(df)

dummies_pclass=pd.get_dummies(data_train['Pclass'],prefix='Pclass')
dummies_pclass.head(3)

dummies_embarked=pd.get_dummies(data_train['Embarked'],prefix='Embarked')
dummies_embarked.loc[61]

dummies_sex=pd.get_dummies(data_train['Sex'],prefix='Sex')
dummies_sex.head(3)

#把处理好的数据合并
df=pd.concat([df,dummies_embarked,dummies_sex,dummies_pclass],axis=1)

#noinspection PyUnresolvedReference
df.drop(['Pclass','Name','Sex','Ticket','Cabin','Embarked'],axis=1,inplace=True)

#选择那些特征作为训练特征
train_df=df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*\|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_df.head(1)

print(train_df.head(1))

#输入模型查看成绩
from abupy import AbuML

train_np=train_df.as_matrix()
y=train_np[:,0]
x=train_np[:,1:]
titanic=AbuML(x,y,train_df)

titanic.estimator.logistic_regression()
titanic.cross_val_accuracy_score()

#逻辑分类是一个线性模型，线性模型就是把特征对应的分类结果的作用相加起来
#特征的非线性的表达式可以分为两类：
#（1）用于表达数值特征本身的非线性因素
#（2）用于表达特征与特征之间存在的非线性关联，并且这种关联关系对分类结果有帮助

#第一种仅适用于数值特征，对应的构造特征的方式有很多种：多项式化和离散化。多项式构造是指将原有数值的高次方作为特征，数据离散化是指将连续的数值划分为一个个区间
#将数值是否在区间内作为特征。高次方让数值内在的表达变得复杂，可描述能力增强，而离散则是让模型来拟合逼近真实的关系描述。

#划分区间
df['Child']=(data_train['Age']<=10).astype(int)
#平方
df['Age*Age']=data_train['Age']*data_train['Age']
#归一化
df['Age*Age_scaled']=scaler.fit_transform(df['Age*Age'])


df['Age*Class']=data_train['Age']*data_train['Pclass']
#归一化
df['Age*Class_scaled']=scaler.fit_transform(df['Age*Class'])

#filter加入新增特征
train_df=df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*\|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*|Child|Age\*Class_.*')
train_df.head(1)

#新增的特征是否能够提升模型表现
train_np=train_df.as_matrix()
y=train_np[:,0]
x=train_np[:,1:]
titanic=AbuML(x,y,train_df)

titanic.estimator.logistic_regression()
titanic.cross_val_accuracy_score()


#一般来说，机器学习中看一个新特征是否发挥作用，最常用的方法就是加进去看模型成绩是否提升，可以同时观察模型给特征分配的权重，看特征发挥作用的大小。
titanic.importances_coef_pd()

#对一些特定的应用场景中，模型的训练非常耗时，可能几天甚至更长，这些应用场景中，需要一些新的数学方法估计新特征是否有效。机器学习中有很多方法评估特征在模型中发挥的作用
#如：
titanic.feature_selection()

#通过人工构造的非线性特征，可以弥补线性模型表达能力的不足，这一手段之所以能生效，背后的原因是：低维的非线性关系可以在高维空间线性展开。
#增加新的特征维度，让分类任务背后的数学表达式变得更加简单，让分类模型更容易挖掘出信息，这是构造新特征有意义的地方，增加特征维度，构造出模型表达不出的内在表达式。
#对于逻辑分类模型而言，这就是通过增加新的非线性特征，完成特征维度的扩展，构造出模型表达不出的非线性的内在关系。

import numpy as np

def score(x, w, b):
    return np.dot(x, w)+b

def softmax(s):
    return np.exp(s) / np.sum(np.exp(s),axis=0)

#y是真实标签，p是预测概率
def cross_entropy(y,p):
    return np.sum(y*np.log(p)+(1-y)*np.log(1-p,axis=1))

#L2正则化
#X是训练样本矩阵，W是权重矩阵，b是偏置向量，y是真实标签矩阵
def loss_func(X,W,b,y):
    C=2  #lmd是一个可调节模型参数
    s=score(X,W,b)
    p=softmax(s)
    return -np.mean(cross_entropy(y,p))+np.mean(np.dot(W.T,W)/C)

#处理过拟合的方法：
#减少特征，降低模型复杂度
#减小调试参数X
#增加训练数据量

#处理前拟合：
#增加特征，增大模型复杂度
#增大调试参数C

#调试模型参数的思路：在训练数据上训练好模型，在测试数据看成绩，将测试集上成绩最好的参数组合作为模型参数，这种思路叫做交叉验证。
#GridSearch是在N-fold Cross-validation基础上的封装实现，可以通过设置一个参数搜索空间暴力搜索所有参数组合，可以同时寻找多个最优参数。

#精确率就是描述当前模型说样本是时，可信的程度。召回率是指描述一种覆盖率，表示模型的预测能抓住这一类别的样本占这一类别全部样本的比例
#在一些场景中精确率和召回率同样重要，就可以那F1分数评估模型的表现

#ROC就是观察模型在不同阈值下，召回率的变化，有意义的地方在于：当测试集中的正负样本的分布变化时，ROC曲线能够保持不变，因此在类别不均衡的数据集中，也可以使用ROC观察模型的表现曲线

#本质上AUC是在模型预测的数据集中，比较正负样本，评估正样本分数排在负样本之上的能力，进而估计模型对正样本预测的可信程度。
#由于AUC指标能够很好地概括不平衡类别的样本集下分类器的性能，因此成为很多机器学习系统中的最终判定标准。


#回归问题和分类问题的区别仅仅在于设定的目标值的类型不同。分类设定的目标值是离散的，意义是类别，而回归设定的目标值是连续的，意义是某种数值。
#线性回归又叫做多项式回归，和逻辑分类类似，都是模型以线性函数描述数据的内在表达式
#简单滴说，线性回归及时y=w*x+b，即在x的数据平面图上找一条线，尽量拟合所有数据点。

titanic.estimator.logistic_regression()
titanic.cross_val_accuracy_score()

titanic.estimator.decision_tree_classifier(criterion='entropy')
#grid search寻找最优决策树
param_grid=dict(max_depth=range(3,10))
best_score_,best_params_=titanic.grid_search_common_clf(param_grid,cv=10,scoring='accuracy')

titanic.estimator.decision_tree_classifier(criterion='entopy',**best_params_)
titanic.cross_val_accuracy_score()

#依赖python的pydot和graphviz包
from sklearn import tree
import pydot
from sklearn.externals.six import StringIO

#为了方便，这里限制决策树的深度观察
titanic.estimator.decision_tree_classifier(criterion='entropy',max_depth=3)
clf=titanic.fit()

#存储树plot
dotfile=StringIO()
tree.export_graphviz(clf,out_file=dotfile,feature_names=titanic.df.cokumns[1:])
pydot.graph_from_dot_data(dotfile.getvalue()).write_png("dtree2.png")
open("dtree2.png")