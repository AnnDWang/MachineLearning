# encoding=utf8　如果不加这一行会报错
from abupy import AbuML

#IRIS花卉数据集
iris=AbuML.create_test_fiter()

#使用Knn
iris.estimator.knn_classifier(n_neighbors=15)

#cross-validation测试
iris.cross_val_accuracy_score()

