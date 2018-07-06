# 西瓜书3.5
# LDA 算法
# 根据书上公式实现
# 看起来好像是对的？

import numpy as np
import matplotlib.pyplot as plt

dataset=np.loadtxt('xigua3.csv',delimiter=',')

# separate the data from the target attributes
X=dataset[:,1:3]
y=dataset[:,3]

m, n = np.shape(X)

goodData=X[:8]
badData=X[8:]

# 计算 均值
u_good=np.mean(goodData,axis=0,keepdims=True)
u_bad=np.mean(badData,axis=0,keepdims=True)

cov_good=(goodData-u_good).T.dot(goodData-u_good)
cov_bad=(badData-u_bad).T.dot(badData-u_bad)

# 类内散度矩阵
Sw=cov_good+cov_bad
# 逆
SwI=np.linalg.inv(Sw)

w=np.dot(SwI,(u_good-u_bad).T)

f3 = plt.figure(3)
plt.xlim(0, 1)
plt.ylim(0, 0.7)

plt.title('watermelon_3a - LDA')
plt.xlabel('density')
plt.ylabel('ratio_sugar')
plt.scatter(X[y == 0, 0], X[y == 0, 1], marker='o', color='k', s=10, label='bad')
plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='o', color='g', s=10, label='good')
plt.legend(loc='upper right')

k = w[1, 0] / w[0, 0]
plt.plot([-1, 1], [-k, k])

for i in range(m):
    curX = (k * X[i, 1] + X[i, 0]) / (1 + k * k)
    if y[i] == 0:
        plt.plot(curX, k * curX, "ko", markersize=3)
    else:
        plt.plot(curX, k * curX, "go", markersize=3)
    plt.plot([curX, X[i, 0]], [k * curX, X[i, 1]], "c--", linewidth=0.3)

plt.show()

a=1

