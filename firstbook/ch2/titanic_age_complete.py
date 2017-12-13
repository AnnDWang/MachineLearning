# -*- coding: utf-8 -*
import pandas as pd

data_train = pd.read_csv("../data/titanic/train.csv")
data_train.info()

import seaborn as sns
# seaborn包含了一系列的统计图形函数

sns.distplot(data_train["Age"].dropna(), kde=True, hist=True)


def set_missing_ages(p_df):
    """均值特征填充"""
    p_df.loc[(p_df.Age.isnull()), 'Age'] = data_train.Age.dropna().mean()
    return p_df

data_train = set_missing_ages(data_train)
data_train_fix1 = set_missing_ages(data_train)
sns.distplot(data_train_fix1["Age"], kde=True, hist=True)

#通过模型预测填充年龄特征
from abupy import AbuML
import sklearn.preprocessing as preprocessing

def set_missing_ages2(p_df):
    """回归模型预测特征填充"""
    age_df = p_df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    # 归一化
    scaler = preprocessing.StandardScaler()
    age_df['Fare_scaled'] = scaler.fit_transform(age_df.Fare.values.reshape(-1, 1))
    del age_df['Fare']
    # 分割已经数据和待预测数据集
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()
    y_inner = known_age[:, 0]
    x_inner = known_age[:, 1:]
    # 训练
    rfr_inner = AbuML(x_inner, y_inner, age_df.Age.notnull())
    rfr_inner.estimator.polynomial_regression(degree=1)
    reg_inner = rfr_inner.fit()
    # 预测
    predicted_ages = reg_inner.predict(unknown_age[:, 1::])
    p_df.loc[(p_df.Age.isnull()), 'Age'] = predicted_ages
    return p_df

data_train = pd.read_csv('../data/titanic/train.csv')
data_train_fix2 = set_missing_ages2(data_train)
sns.distplot(data_train_fix2["Age"], kde=True, hist=True)

# 处理cabin特征
def set_cabin_type(p_df):
    p_df.loc[(p_df.Cabin.notnull()), 'Cabin'] = "Yes"
    p_df.loc[(p_df.Cabin.isnull()), 'Cabin'] = "No"
    return p_df

def train_val(data):
    """封装所有处理训练步骤"""
    # 处理离散特征
    dummies__cabin = pd.get_dummies(data['Cabin'], prefix='Cabin')
    dummies__embarked = pd.get_dummies(data['Embarked'], prefix='Embarked')
    dummies__sex = pd.get_dummies(data['Sex'], prefix='Sex')
    dummies__pclass = pd.get_dummies(data['Pclass'], prefix='Pclass')
    df = pd.concat([data, dummies__cabin, dummies__embarked, dummies__sex, dummies__pclass], axis=1)
    df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
    # 归一化数据
    scaler = preprocessing.StandardScaler()
    df['Age_scaled'] = scaler.fit_transform(df['Age'].values.reshape(-1, 1))
    df['Fare_scaled'] = scaler.fit_transform(df['Fare'].values.reshape(-1, 1))
    df['SibSp_scaled'] = scaler.fit_transform(df['SibSp'].astype(float).values.reshape(-1, 1))
    df['Parch_scaled'] = scaler.fit_transform(df['Parch'].astype(float).values.reshape(-1, 1))
    # 选择特征
    train_df = df.filter(regex='Survived|Age_.*|SibSp_.*|Parch_.*|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    train_np = train_df.as_matrix()
    y = train_np[:, 0]
    x = train_np[:, 1:]
    titanic = AbuML(x, y, train_df)
    titanic.estimator.logistic_classifier()
    titanic.cross_val_accuracy_score()

data_train_fix1 = set_cabin_type(data_train_fix1)
train_val(data_train_fix1)

data_train_fix2 = set_cabin_type(data_train_fix2)
train_val(data_train_fix2)


