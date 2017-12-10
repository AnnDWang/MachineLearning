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