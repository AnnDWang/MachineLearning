#

import numpy as np
import matplotlib.pyplot as plt

# 西瓜书 3.3 对率回归

def sigmoid(x):
    y=1.0/(1+np.exp(-x))
    return y


def likelihood_sub(x, y, beta):
    return -y * np.dot(beta, x.T) + np.math.log(1 + np.math.exp(np.dot(beta, x.T)))


def likelihood(X, y, beta):
    sum = 0
    m, n = np.shape(X)

    for i in range(m):
        sum += likelihood_sub(X[i], y[i], beta)

    return sum


def gradDscent_1(X, y):
    h = 0.1  # step length of iterator
    max_times = 500
    m, n = np.shape(X)

    beta = np.zeros(n)  # parameter and initial
    delta_beta = np.ones(n) * h

    llh = 0
    llh_temp = 0
    for i in range(max_times):
        beta_temp = beta
        for j in range(n):
            # for partial derivative
            beta[j] += delta_beta[j]
            llh_temp = likelihood(X, y, beta)
            delta_beta[j] = -h * (llh_temp - llh) / delta_beta[j]

        beta = beta_temp + delta_beta
        llh = likelihood(X, y, beta)
    return beta


def gradDscent_2(X, y):
    m, n = np.shape(X)
    h = 0.5  # step length of iterator and initial
    beta = np.zeros(n)  # parameter and initial
    delta_beta = np.ones(n) * h
    llh = 0
    llh_tmp = 0
    b = np.zeros((n, m))  # for show convergence curve of parameter

    for i in range(m):
        beta_temp = beta
        for j in range(n):
            # for partial derivative
            h = 05. * 1 / (1 + i + j)  # change the step length of iterator
            beta[j] += delta_beta[j]

            b[j, i] = beta[j]

            llh_tmp = likelihood_sub(X[i], y[i], beta)
            delta_beta[j] = -h * (llh_tmp - llh) / delta_beta[j]

            beta[j] = beta_temp[j]

        beta += delta_beta
        llh = likelihood_sub(X[i], y[i], beta)

    t = np.arange(m)

    f2 = plt.figure(3)

    p1 = plt.subplot(311)
    p1.plot(t, b[0])
    plt.ylabel('w1')

    p2 = plt.subplot(312)
    p2.plot(t, b[1])
    plt.ylabel('w2')

    p3 = plt.subplot(313)
    p3.plot(t, b[2])
    plt.ylabel('b')

    plt.show()

    return beta

# 自己根据书上59-60页的牛顿法的更新公式的方法
def newton(X,y):
    max_times = 500
    m, n = np.shape(X)
    N = X.shape[0]
    lr=0.05
    beta = np.zeros(n)  # parameter and initial

    z=X.dot(beta.T)
    for i in range(max_times):
        # 计算牛顿法更新的分子
        p1=np.exp(z)/(1+np.exp(z))
        fenzi=0.0
        fenmu=0.0
        for j in range(m):
            fenzi=fenzi-np.dot(X[j].T,y[j]-p1[j])
            fenmu=fenmu+np.dot(X[j],X[j].T)*p1[j]*(1-p1[j])
        beta=beta-fenzi/fenmu

    return beta




    l = np.sum(-y * z + np.log(1 + np.exp(z)))
    print(l)
    return beta


def sigmoid(x,beta):
    return 1.0 / (1 + np.math.exp(- np.dot(beta.T, x)))

def predict(X, beta):
    m, n = np.shape(X)
    y = np.zeros(m)

    for i in range(m):
        if sigmoid(X[i], beta) > 0.5: y[i] = 1;
    return y


# load the csv file as a numpy as matrix
# separate the data with "" blank ,\t

dataset=np.loadtxt('xigua3.csv',delimiter=',')

# separate the data from the target attributes
X=dataset[:,1:3]
y=dataset[:,3]

goodData=dataset[:8]
badData=dataset[8:]

# return the size

m,n=np.shape(X)

print(m,n)

# draw scatter diagram to show the raw data

f1=plt.figure(1)
plt.title('xigua_3a')
plt.xlabel('density')
plt.ylabel('ratio_sugar')

# a=goodData[:,1]
# b=goodData[:,2]
#
# c=X[y == 0, 0]
# d=X[y == 0, 1]
#
# plt.scatter(goodData[:,1], goodData[:,2], marker='o', color='g', s=100, label='good')
# plt.scatter(X[y == 0, 0], X[y == 0, 1], marker='o', color='k', s=100, label='bad')
#
# plt.legend()
#
# plt.show()

# from sklearn import  metrics
# from sklearn import model_selection
# from sklearn.linear_model import  LogisticRegression
# import matplotlib.pylab as pl
#
# X_train,X_test,y_train,y_test=model_selection.train_test_split(X,y,test_size=0.5,random_state=0)
#
# log_model=LogisticRegression()
# log_model.fit(X_train,y_train)
#
# y_pred=log_model.predict(X_test)
#
# print(metrics.confusion_matrix(y_test,y_pred))
# print(metrics.classification_report(y_test,y_pred))
#
#
# precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_pred)

from sklearn import model_selection


m,n=np.shape(X)
X_ex=np.c_[X,np.ones(m)]

a=1

X_train,X_test,y_train,y_test=model_selection.train_test_split(X_ex,y,test_size=0.5,random_state=0)

beta=newton(X_train,y_train)

y_pred=predict(X_test,beta)

m_test=np.shape(X_test)[0]

cfmat = np.zeros((2, 2))
for i in range(m_test):
    if y_pred[i] == y_test[i] == 0:
        cfmat[0, 0] += 1
    elif y_pred[i] == y_test[i] == 1:
        cfmat[1, 1] += 1
    elif y_pred[i] == 0:
        cfmat[1, 0] += 1
    elif y_pred[i] == 1:
        cfmat[0, 1] += 1

print(cfmat)





