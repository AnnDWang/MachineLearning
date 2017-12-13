# -*- coding: utf-8 -*
# 不同k值下的分类效果比较
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier

# 获取iris数据集
iris = datasets.load_iris()

# create X and y
X = iris.data
y = iris.target

n_neighbors = [1, 15, 50]

X_plot = X[:, :2]  # 只取前两维特征，方便plot

h = .02  # plot网格单位长

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

for k in n_neighbors:
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_plot, y)
    y_pred = clf.predict(X_plot)

    x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
    y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # 用 color plot 观察分类结果
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # plot训练样本点
    plt.scatter(X_plot[:, 0], X_plot[:, 1], c=y, cmap=cmap_bold)
    plt.title("3-Class classification (K={})".format(k))
    plt.axis('tight')
    # plt.savefig('knn_k_{}.png'.format(k))

plt.show()
