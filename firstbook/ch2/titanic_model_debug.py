# -*- coding: utf-8 -*
from abupy import AbuML

# create_test_more_fiter用于快速测试API，封装了titanic数据集合相关的特征处理
titanic = AbuML.create_test_more_fiter()
titanic.estimator.logistic_classifier()

# 学习曲线
titanic.plot_learning_curve()