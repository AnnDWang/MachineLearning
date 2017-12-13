# -*- coding: utf-8 -*
from sklearn import datasets
from sklearn.cross_validation import  train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from abupy import AbuML

#数据集
scikit_boston=datasets.load_boston()
x=scikit_boston.data
y=scikit_boston.target
df=pd.DataFrame(data=np.c_[x,y])
columns=np.append(scikit_boston.feature_names,['MEDV'])
df.head(1)

#检查缺失
df.info()

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

#归一化数据
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.fit_transform(x_test)

#模型训练
df=pd.DataFrame(data=np.c_[x_train,y_train])
columns=np.append(scikit_boston.feature_names,['MEDV'])
boston=AbuML(x_train,y_train,df)
boston.estimator.polynomial_regression(degree=1)
reg=boston.fit()

#测试集上预测
y_pred=reg.predict(x_test)

from sklearn.metrics import  r2_score
r2_score(y_test,y_pred)

#平方展开
boston.estimator.polynomial_regression(degree=2)

reg=boston.fit()

y_pred=reg.predict(x_test)
r2_score(y_test,y_pred)

# IRIS花卉数据集
iris = AbuML.create_test_fiter()

# 使用KNN
iris.estimator.knn_classifier()

from abupy import KFold

kf = KFold(len(iris.y), n_folds=10, shuffle=True)

for train_index, test_index in kf:
    x_train, x_test = iris.x[train_index], iris.x[test_index]
    y_train, y_test = iris.y[train_index], iris.y[test_index]

x_train.shape, x_test.shape, y_train.shape, y_test.shape

from abupy import ABuMLGrid
# 定义参数搜索范围
_, best_params = ABuMLGrid.grid_search_init_kwargs(iris.estimator.clf, iris.x, iris.y, scoring='accuracy',
                                             param_name='n_neighbors',param_range=range(1, 31), show=True)
print best_params