# -*- coding: utf-8 -*
#代码来自于《机器学习之路》
import numpy as np # numpy:快速操作结构数组的工具
import pandas as pd # pandas:数据分析处理工具
#import matplotlib.pyplot as lpt #matplotlib画图工具
from sklearn import datasets # datasets :sklearn 的示例数据集

# 数据集 0-setosa、1-versicolor、2-virginica
scikit_iris = datasets.load_iris()
# 转换成pandas的DataFrame数据格式，方便观察数据
iris = pd.DataFrame(
    data=np.c_[scikit_iris['data'],scikit_iris['target']],
    columns=np.append(scikit_iris.feature_names,['y'])
)

# 观察数据格式
print iris.head(2)

# 检查数据是否有缺失
print iris.isnull().sum()

# 观察样本中按类别数量是否比较均衡
print iris.groupby('y').count()

# 选择全部特征训练模型
X = iris[scikit_iris.feature_names]
# label
y = iris['y']

# 第一步，选择model
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1)
# 第二步，fit X，y
knn.fit(X,y)
# 第三步，predict新数据
print knn.predict([[3,2,2,5]])


# 分割训练集和测试集
from sklearn.model_selection import train_test_split
from sklearn import metrics

# 分割训练-测试集
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=4)
# K=15
knn=KNeighborsClassifier(n_neighbors=15)
knn.fit(X_train,y_train)

y_pred_on_train = knn.predict(X_train)
y_pred_on_test=knn.predict(X_test)

#print metrics.accuracy_score(y_train,y_pred_on_train)
print('accuracy: ：{}'.format(metrics.accuracy_score(y_test, y_pred_on_test)))