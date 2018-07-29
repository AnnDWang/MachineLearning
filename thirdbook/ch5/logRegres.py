from numpy import *

# logistic 回归梯度上升优化算法
def loadDataSet():
    dataMat=[]
    labelMat=[]
    fr=open('testSet.txt')
    for line in fr.readlines():
        lineArr=line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

def gradAscent(dataMatIn,classLabels):
    dataMatrix=mat(dataMatIn)
    labelMat=mat(classLabels).transpose()
    m,n=shape(dataMatrix)
    alpha=0.001
    maxCycles=500
    weights=ones((n,1))
    for k in range(maxCycles):
        h=sigmoid(dataMatrix*weights)
        error=(labelMat-h)
        weights=weights+alpha*dataMatrix.transpose()*error
    return weights

dataArr,labelMat=loadDataSet()
weights=gradAscent(dataArr,labelMat)

import matplotlib.pyplot as plt

# 画出数据集合logistic回归最佳拟合直线的函数
def plotBestFit(wei):
    weights=wei
    dataMat,labelMat=loadDataSet()
    dataArr=array(dataMat)
    n=shape(dataArr)[0]
    xcord1=[]
    ycord1=[]
    xcord2=[]
    ycord2=[]
    for i in range(n):
        if int(labelMat[i])==1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x=arange(-3.0,3.0,0.1)
    y=(-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

# 随机梯度上升算法
def stocGradAscent0(dataMatrix,classLabels):
    m,n=shape(dataMatrix)
    alpha=0.01
    weights=ones(n)
    for i in range(m):
        h=sigmoid(sum(dataMatrix[i]*weights))
        error=classLabels[i]-h
        weights=weights+alpha*error*dataMatrix[i]
    return weights

weights=stocGradAscent0(array(dataArr),labelMat)
#

# 改进的随机梯度上升算法
# alpha在每次迭代时候都会调整，这回缓解数据波动或者高频波动
# 虽然alpha会随着迭代次数不断减小，但永远不会减小到0
# 这是因为alpha更新公式中存在一个常数项，保证多次迭代之后新数据仍然有一定的影响
# 如果要处理的问题是动态变化的，那么可以适当加大常数项，来确保新的值获得更大的回归系数
# 在降低alpha函数中，每次减少1/(j+i)，其中j是迭代次数，i是样本点的下标，
# 这样当j<<max(i)时，alpha就不是严格下降的
# 避免参数的严格下降也常见于模拟退火等其他优化算法
def stocGradAscent1(dataMatrix,classLabels,numIter=150):
    m,n=shape(dataMatrix)
    weights=ones(n)
    for j in range(numIter):
        dataIndex=list(range(m))
        for i in range(m):
            alpha=4/(1.0+j+i)+0.01 # alpha在每次迭代时候都会调整
            randIndex=int(random.uniform(0,len(dataIndex))) # 随机选取更新
            h=sigmoid(sum(dataMatrix[randIndex]*weights))
            error=classLabels[randIndex]-h
            weights=weights+alpha*error*dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

weights=stocGradAscent1(array(dataArr),labelMat)

# plotBestFit(weights)

#logistic回归分类函数
def classifyVector(inX,weights):
    prob=sigmoid(sum(inX*weights))
    if prob>0.5:return 1.0
    else: return 0.0

def colicTest():
    frTrain=open('horseColicTraining.txt')
    frTest=open('horseColicTest.txt')
    trainingSet=[]
    trainingLabels=[]
    for line in frTrain.readlines():
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights=stocGradAscent1(array(trainingSet),trainingLabels,500)
    errorCount=0
    numTestVec=0.0
    for line in frTest.readlines():
        numTestVec+=1.0
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr),trainWeights))!=int(currLine[21]):
            errorCount+=1
    errorRate=(float(errorCount)/numTestVec)
    print('the error rate of this test is : %f'% errorRate)
    return errorRate

def multiTest():
    numTests=10
    errorSum=0.0
    for k in range(numTests):
        errorSum+=colicTest()
    print('after %d iterations the average error rate is: %f' %(numTests,errorSum/float(numTests)))

multiTest()