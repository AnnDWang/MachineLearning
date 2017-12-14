# -*- coding: utf-8 -*
from abupy import AbuML
import sklearn.preprocessing as preprocessing
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def set_missing_ages(p_df):
    p_df.loc[(p_df.Age.isnull()), 'Age'] = data_train.Age.dropna().mean()
    return p_df

# 处理cabin特征
def set_cabin_type(p_df):
    p_df.loc[(p_df.Cabin.notnull()), 'Cabin'] = "Yes"
    p_df.loc[(p_df.Cabin.isnull()), 'Cabin'] = "No"
    return p_df

data_train = pd.read_csv('../data/titanic/train.csv')
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

from abupy import ABuMLGrid

# # 决策树
# titanic.estimator.decision_tree_classifier()
# # grid seach寻找最优的决策树层数
# best_score_, best_params_ = ABuMLGrid.grid_search_init_kwargs(titanic.estimator.clf, titanic.x, titanic.y,
#                                                         param_name='max_depth',param_range=range(3, 10), show=True)
#
# titanic.estimator.decision_tree_classifier(**best_params_)
# titanic.cross_val_accuracy_score()

#随机森林
titanic.estimator.random_forest_classifier()

#grid search 寻找最优参数，n_estimators个体模型数量
#max_features 特征自己样本比例，max_depth层数深度

param_grid={
    'n_estimators':range(80,150,10),
    'max_features':np.arange(.5,1.,.1).tolist(),
    'max_depth':range(1,10)+[None]
}
#n_jobs=-1开启多线程
best_score_, best_params_ = ABuMLGrid.grid_search_mul_init_kwargs(titanic.estimator.clf, titanic.x, titanic.y,
                                                                       param_grid=param_grid, show=True, n_jobs=-1)

best_score_, best_params_

titanic.estimator.random_forest_classifier(**best_params_)
titanic.cross_val_accuracy_score()

# GBDT
titanic.estimator.xgb_classifier()

# grid seach寻找最优的参数：n_estimators个体模型数量；max_depth层数深度
param_grid = {
    'n_estimators': range(80, 150, 10),
    'max_depth': range(1, 10)
}

# n_jobs=-1开启多线程
best_score_, best_params_ = ABuMLGrid.grid_search_mul_init_kwargs(titanic.estimator.clf, titanic.x, titanic.y,
                                                                       param_grid=param_grid, show=True, n_jobs=-1)
best_score_, best_params_

titanic.estimator.xgb_classifier(**best_params_)
titanic.cross_val_accuracy_score()

# 逻辑分类
titanic.estimator.logistic_classifier()
titanic.cross_val_accuracy_score()

# 随机森林
param = {'max_depth': 8, 'max_features': 0.6, 'n_estimators': 80}
titanic.estimator.random_forest_classifier(**param)
titanic.cross_val_accuracy_score()

# GBDT
param = {'max_depth': 5, 'n_estimators': 140}
titanic.estimator.xgb_classifier(**param)
titanic.cross_val_accuracy_score()

# 准备训练好的模型
titanic.estimator.logistic_classifier()
lr = titanic.fit()
param = {'max_depth': 8, 'max_features': 0.6, 'n_estimators': 80}
titanic.estimator.random_forest_classifier(**param)
rf = titanic.fit()
param = {'max_depth': 5, 'n_estimators': 140}
titanic.estimator.xgb_classifier(**param)
gbdt = titanic.fit()

# 构造stacking训练集，融合三个模型的预测的概率值作为特征数据
x_stk = np.array([lr.predict_proba(x)[:, 0], rf.predict_proba(x)[:, 0], gbdt.predict_proba(x)[:, 0]]).T
x_df_stk = pd.DataFrame(x_stk, columns=['lr', 'rf', 'gbdt'])
y_df = pd.DataFrame(y, columns=['y'])
df = y_df.join(x_df_stk)

# stacking模型
stackings = AbuML(x_stk, y, df)

stackings.estimator.logistic_classifier()

# 获得titanic的融合模型stk
stk = stackings.fit()

stackings.cross_val_accuracy_score()

from abupy import KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics


def lr_model(x_train, x_test, y_train, y_test):
    """返回训练好的逻辑分类模型及分数"""
    lr = LogisticRegression(C=1.0)
    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_test)
    return lr, metrics.accuracy_score(y_test, y_pred)


def rf_model(x_train, x_test, y_train, y_test):
    """返回训练好的随机森林模型及分数"""
    param_grid = {
        'n_estimators': range(80, 120, 10),
        'max_features': np.arange(.6, .9, .1).tolist(),
        'max_depth': list(range(3, 9)) + [None]
    }

    grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=10, scoring='accuracy', n_jobs=-1)
    grid.fit(x_train, y_train)
    rf = RandomForestClassifier(**grid.best_params_)
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)
    return rf, metrics.accuracy_score(y_test, y_pred)


def gbdt_model(x_train, x_test, y_train, y_test):
    """返回训练好的GBDT模型及分数"""
    param_grid = {
        'n_estimators': list(range(80, 120, 10)),
        'max_features': np.arange(.6, .9, .1).tolist(),
        'max_depth': list(range(3, 9)) + [None]
    }

    grid = GridSearchCV(GradientBoostingClassifier(), param_grid, cv=10, scoring='accuracy', n_jobs=-1)
    grid.fit(x_train, y_train)
    gbdt = GradientBoostingClassifier(**grid.best_params_)
    gbdt.fit(x_train, y_train)
    y_pred = gbdt.predict(x_test)
    return gbdt, metrics.accuracy_score(y_test, y_pred)


def stack_models(x_train, x_test, y_train, y_test):
    """返回融合后的模型及分数"""
    param_grid = {
        'C': [.01, .1, 1, 10]
    }
    grid = GridSearchCV(LogisticRegression(), param_grid, cv=10, scoring='accuracy', n_jobs=-1)
    grid.fit(x_train, y_train)
    stk = LogisticRegression(penalty='l1', tol=1e-6, **grid.best_params_)
    stk.fit(x_train, y_train)
    y_pred = stk.predict(x_test)
    return rf, metrics.accuracy_score(y_test, y_pred)


kf = KFold(len(titanic.y), n_folds=5, shuffle=True)
lr_scores = []
rf_scores = []
gbdt_scores = []
stk_scores = []

for train_index, test_index in kf:
    x_train, x_test = titanic.x[train_index], titanic.x[test_index]
    y_train, y_test = titanic.y[train_index], titanic.y[test_index]

    # 单个模型成绩
    lr, lr_score = lr_model(x_train, x_test, y_train, y_test)
    rf, rf_score = rf_model(x_train, x_test, y_train, y_test)
    gbdt, gbdt_score = gbdt_model(x_train, x_test, y_train, y_test)

    # stacking
    x_train_stk = np.array(
        [lr.predict_proba(x_train)[:, 0], rf.predict_proba(x_train)[:, 0], gbdt.predict_proba(x_train)[:, 0]]).T
    x_test_stk = np.array(
        [lr.predict_proba(x_test)[:, 0], rf.predict_proba(x_test)[:, 0], gbdt.predict_proba(x_test)[:, 0]]).T
    stk, stk_score = stack_models(x_train_stk, x_test_stk, y_train, y_test)

    # append score
    lr_scores.append(lr_score)
    rf_scores.append(rf_score)
    gbdt_scores.append(gbdt_score)
    stk_scores.append(stk_score)

print('lr mean score: {}'.format(np.mean(lr_scores)))
print('rf mean score: {}'.format(np.mean(rf_scores)))
print('gbdt mean score: {}'.format(np.mean(gbdt_scores)))
print('stk mean score: {}'.format(np.mean(stk_scores)))


print('lr std score: {}'.format(np.std(lr_scores)))
print('rf std score: {}'.format(np.std(rf_scores)))
print('gbdt std score: {}'.format(np.std(gbdt_scores)))
print('stk std score: {}'.format(np.std(stk_scores)))


import numpy as np

def three_kfolder(data, n_folds=5, shuffle=True, ratios=[4, 1, 2]):
    """按ratios数组随机(shuffle)三分割数据集，返回：traing_set, stacking_set, testing_set"""
    assert ratios and len(ratios) == 3, 'ratios必须是3-items-arraylike数组'
    data = np.array(data)
    N = len(data)
    ratios_nor = np.array(ratios).astype(float) / np.sum(ratios)
    ratios_num = (ratios_nor * N).astype(int).cumsum()

    for i in range(n_folds):
        ind = list(range(len(data)))
        np.random.shuffle(ind)
        data_shuf = data[ind]
        yield data_shuf[:ratios_num[0]], data_shuf[ratios_num[0]:ratios_num[1]], data_shuf[ratios_num[1]:ratios_num[2]]

# 使用demo
data = ['a', 'c', 'd', 'e', 'g', '2','f', 'c','3', 'p']
for traing_set, stacking_set, testing_set in three_kfolder(data):
    print(traing_set)
    print(stacking_set)
    print(testing_set)



