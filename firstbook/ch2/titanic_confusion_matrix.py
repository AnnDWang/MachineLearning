# -*- coding: utf-8 -*
from abupy import AbuML

# 泰坦尼克号生存预测
titanic = AbuML.create_test_more_fiter()

titanic.plot_confusion_matrices()

from abupy import ABuMLExecute
from sklearn import metrics

titanic_y_pred = ABuMLExecute.run_cv_estimator(titanic.get_fiter(), titanic.x, titanic.y, n_folds=10)
confusion_matrix = metrics.confusion_matrix(titanic.y, titanic_y_pred)
TP = confusion_matrix[1, 1]
TN = confusion_matrix[0, 0]
FP = confusion_matrix[0, 1]
FN = confusion_matrix[1, 0]
print TP, TN, FP, FN

assert metrics.accuracy_score(titanic.y, titanic_y_pred) == (TP + TN) / float(TP + TN + FP + FN)

# “生存”类别的精确率
tit_precision = TP / float(TP + FP)
# “生存”类别的召回率
tit_recall = TP / float(TP + FN)

assert metrics.precision_score(titanic.y, titanic_y_pred) == tit_precision
assert metrics.recall_score(titanic.y, titanic_y_pred) == tit_recall

assert metrics.f1_score(titanic.y, titanic_y_pred) == 2 * tit_precision * tit_recall / (tit_precision + tit_recall)

# IRIS花卉数据集
iris = AbuML.create_test_fiter()
iris.estimator.knn_classifier(n_neighbors=20)

confusion = iris.plot_confusion_matrices()

iris_y_pred = ABuMLExecute.run_cv_estimator(iris.get_fiter(), iris.x, iris.y, n_folds=10)
# precision
metrics.precision_score(iris.y, iris_y_pred, average=None)

# recall
metrics.recall_score(iris.y, iris_y_pred, average=None)

# f1-score
metrics.f1_score(iris.y, iris_y_pred, average=None)

print(metrics.precision_score(iris.y, iris_y_pred, average=None))
print(metrics.recall_score(iris.y, iris_y_pred, average='macro'))
print(metrics.f1_score(iris.y, iris_y_pred, average='weighted'))

# K-folder下的ROC曲线和AUC面积（area）
titanic.plot_roc_estimator()