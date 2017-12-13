# -*- coding: utf-8 -*
import numpy as np

def entropy(P):
    """根据每个样本出现的概率，计算信息量，输入P是数据集上每个数值统计的频率（概率）向量"""
    return -np.sum(P * np.log2(P))

from abupy import AbuML
import sklearn.preprocessing as preprocessing
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def set_missing_ages(p_df):
    age_df = p_df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    # 归一化
    scaler = preprocessing.StandardScaler()
    age_df['Fare_scaled'] = scaler.fit_transform(age_df['Fare'].values.reshape(-1, 1))
    del age_df['Fare']
    # 分割已经数据和待预测数据集
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()
    y_inner = known_age[:, 0]
    x_inner = known_age[:, 1:]
    rfr_inner = AbuML(x_inner, y_inner, age_df.Age.notnull())
    rfr_inner.estimator.polynomial_regression(degree=1)
    reg_inner = rfr_inner.fit()
    predicted_ages = reg_inner.predict(unknown_age[:, 1::])
    p_df.loc[(p_df.Age.isnull()), 'Age'] = predicted_ages
    return p_df

# 处理cabin特征
def set_cabin_type(p_df):
    p_df.loc[(p_df.Cabin.notnull()), 'Cabin'] = "Yes"
    p_df.loc[(p_df.Cabin.isnull()), 'Cabin'] = "No"
    return p_df

data_train = pd.read_csv('../data/titanic/train.csv')
# 处理数据缺失
data_train = set_missing_ages(data_train)
data_train = set_cabin_type(data_train)

# 处理离散特征
dummies__cabin = pd.get_dummies(data_train['Cabin'], prefix='Cabin')
dummies__embarked = pd.get_dummies(data_train['Embarked'], prefix='Embarked')
dummies__sex = pd.get_dummies(data_train['Sex'], prefix='Sex')
dummies__pclass = pd.get_dummies(data_train['Pclass'], prefix='Pclass')
df = pd.concat([data_train, dummies__cabin, dummies__embarked, dummies__sex, dummies__pclass], axis=1)
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

from abupy import ABuMLGrid
# 切换决策树
titanic.estimator.decision_tree_classifier(criterion='entropy')
# grid seach寻找最优的决策树层数
best_score_, best_params_ = ABuMLGrid.grid_search_init_kwargs(titanic.estimator.clf, titanic.x, titanic.y,
                                                        param_name='max_depth',param_range=range(3, 10), show=True)
best_score_, best_params_

from sklearn import tree
from sklearn.externals.six import StringIO

try:
    import pydot
    # 为了方便试图，这里限制决策树的深度观察
    titanic.estimator.decision_tree_classifier(criterion='entropy', max_depth=3)
    clf = titanic.fit()

    # 存储树plot
    dotfile = StringIO()
    tree.export_graphviz(clf, out_file=dotfile, feature_names=titanic.df.columns[1:])
    pydot.graph_from_dot_data(dotfile.getvalue()).write_png("dtree2.png")

except ImportError:
    print('这段代码依赖python的pydot和graphviz包')

##图片没有出来