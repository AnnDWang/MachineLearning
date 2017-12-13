# -*- coding: utf-8 -*
import pandas as pd

data_train=pd.read_csv("../data/titanic/train.csv")
data_train.info()

import seaborn as sns #seaborn包含了一系列的统计图形函数

sns.distplot(data_train["Age"].dropna(),kde=True,hist=True)


#按均值填充
def set_missing_ages(p_df):
    """均值特征填充"""
    p_df.loc[(p_df.Age.isnull()),'Age']=data_train.Age.dropna().mean()
    return p_df

data_train=set_missing_ages(data_train)

data_train_fix1=set_missing_ages(data_train)

sns.distplot(data_train_fix1["Age"],kde=True,hist=True)

#通过模型预测填充年龄特征
from abupy import AbuML
import sklearn.preprocessing as preprocessing

def set_missing_ages2(p_df):
    age_df=p_df[['Age','Fare','Parch','SibSp','Pclass']]
    #归一化
    scaler=preprocessing.StandardScaler()
    age_df['Fare_scaled']=scaler.fit_transform(age_df['Fare'])
    del age_df['Fare']

    #分割已有数据和带预测数据集
    known_age=age_df[age_df.Age.notnull()].as_matrix()
    unknown_age=age_df[age_df.Age.isnull()].as_matrix()
    y_inner=known_age[:,0]
    x_inner=known_age[:,1:]
    #训练
    rfr_inner=AbuML(x_inner,y_inner,age_df.Age.notnull())
    rfr_inner.estimator.polynomial_regression(degree=1)
    reg_inner=rfr_inner.fit()

    #预测
    predicted_ages=reg_inner.predict(unknown_age[:,1::])
    p_df.loc[(p_df.Age.isnull()),'Age']=predicted_ages
    return  p_df

data_train=pd.read_csv("../data/titanic/train.csv")
data_train_fix2=set_missing_ages2(data_train)
sns.distplot(data_train_fix2["Age"],ked=True,hist=True)