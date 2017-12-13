# encoding=utf8　如果不加这一行会报错
#代码来自于书1.4.3小节
#暂时没有成功将abupy安装到anaconda环境中

from abupy import AbuML

#IRIS 花卉数据集

iris=AbuML.create_test_fiter()

#使用逻辑分类，损失函数指定为交叉熵
iris.estimator.logistic_classifier(multi_class='multinomial',solver='lbfgs')

#cross_validation测试
iris.cross_val_accuracy_score()

