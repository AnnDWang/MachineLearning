from numpy import *

def loadDataSet(fileName):
    numFeat=len(open(fileName).readline().split('\t'))-1
    dataMat=[]
    labelMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        lineArr=[]
        curLine=line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def standRgres(xArr,yArr):
    xMat=mat(xArr)
    yMat=mat(yArr).T
    xTx=xMat.T*xMat
    if linalg.det(xTx)==0.0:
        print('the matrix is singular, cannot do inverse')
        return
    ws=xTx.I*(xMat.T*yMat)
    return ws

xArr,yArr=loadDataSet('ex0.txt')
ws=standRgres(xArr,yArr)
print(ws)
xMat=mat(xArr)
yMat=mat(yArr)
yHat=xMat*ws

# 绘制出数据集散点图和最佳啮合直线图
import  matplotlib.pyplot as plt
# fig=plt.figure()
# ax=fig.add_subplot(111)
# ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0])
# xCopy=xMat.copy()
# xCopy.sort(0)
# yHat=xCopy*ws
# ax.plot(xCopy[:,1],yHat)
# plt.show()


# 局部加权线性回归函数
def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat=mat(xArr)
    yMat=mat(yArr).T
    m=shape(xMat)[0]
    weights=mat(eye(m)) # 创建对角矩阵
    for j in range(m):
        diffMat=testPoint-xMat[j,:]
        # 权重大小以指数级衰减
        weights[j,j]=exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx=xMat.T*(weights*xMat)
    if linalg.det(xTx)==0.0:
        print('this matrix is singular, cannot do inverse')
        return
    ws=xTx.I*(xMat.T*(weights*yMat))
    return testPoint*ws

def lwlrTest(testArr,xArr,yArr,k=1.0):
    m=shape(testArr)[0]
    yHat=zeros(m)
    for i in range(m):
        yHat[i]=lwlr(testArr[i],xArr,yArr,k)
    return yHat

xArr,yArr=loadDataSet('ex0.txt')
print(lwlr(xArr[0],xArr,yArr,1.0))
print(lwlr(xArr[0],xArr,yArr,0.001))
yHat=lwlrTest(xArr,xArr,yArr,0.003)
xMat=mat(xArr)
srtInd=xMat[:,1].argsort(0)
xSort=xMat[srtInd][:,0,:]

# fig=plt.figure()
# ax=fig.add_subplot(111)
# ax.plot(xSort[:,1],yHat[srtInd])
# ax.scatter(xMat[:,1].flatten().A[0],mat(yArr).T.flatten().A[0],s=2,c='red')
# plt.show()

# 预测鲍鱼的年龄
def rssError(yArr,yHatArr):
    return ((yArr-yHatArr)**2).sum()

abX,abY=loadDataSet('abalone.txt')
yHat01=lwlrTest(abX[0:99],abX[0:99],abY[0:99],0.1)
yHat1=lwlrTest(abX[0:99],abX[0:99],abY[0:99],1)
yHat10=lwlrTest(abX[0:99],abX[0:99],abY[0:99],10)

# print(rssError(abY[0:99],yHat01.T))
# print(rssError(abY[0:99],yHat1.T))
# print(rssError(abY[0:99],yHat10.T))

# 岭回归

# 用于计算回归系数，
# 实现了给定了lambda下的岭回归求解
# 如果没有指定lambda，默认为0.2
def ridgeRegres(xMat,yMat,lam=0.2):
    xTx=xMat.T*xMat
    denom=xTx+eye(shape(xMat)[1])*lam
    if linalg.det(denom)==0.0:
        print('this matrix is singular, cannot do inverse')
        return
    ws=denom.I*(xMat.T*yMat)
    return ws

# 用于在一组lambda上测试结果
# 为了使用岭回归和缩减技术，首先需要对特征进行标准化处理
# 具体的做法是所有特征都减去各自的均值并处以方差
def ridgeTest(xArr,yArr):
    xMat=mat(xArr)
    yMat=mat(yArr).T
    yMean=mean(yMat,0)
    yMat=yMat-yMean
    xMeans=mean(xMat,0)
    xVar=var(xMat,0)
    xMat=(xMat-xMeans)/xVar
    numTestPts=30
    wMat=zeros((numTestPts,shape(xMat)[1]))
    for i in range(numTestPts):
        ws=ridgeRegres(xMat,yMat,exp(i-10))
        wMat[i,:]=ws.T
    return wMat
abX,abY=loadDataSet('abalone.txt')
ridgeWeights=ridgeTest(abX,abY)

# fig=plt.figure()
# ax=fig.add_subplot(111)
# ax.plot(ridgeWeights)
# plt.show()

def regularize(xMat):#regularize by columns
    inMat = xMat.copy()
    inMeans = mean(inMat,0)   #calc mean then subtract it off
    inVar = var(inMat,0)      #calc variance of Xi then divide by it
    inMat = (inMat - inMeans)/inVar
    return inMat

# 前向逐步线性回归
# EPS表示每次迭代需要调整的步长
# numIt表示迭代次数
def stageWise(xArr,yArr,eps=0.01,numIt=100):
    xMat=mat(xArr)
    yMat=mat(yArr).T
    yMean=mean(yMat,0)
    yMat=yMat-yMean
    xMat=regularize(xMat)
    m,n=shape(xMat)
    returnMat=zeros((numIt,n))
    ws=zeros((n,1))
    wsTest=ws.copy()
    wsMax=ws.copy()
    for i in range(numIt):
        print(ws.T)
        lowestError=inf
        for j in range(n):
            for sign in [-1,1]:
                wsTest=ws.copy()
                wsTest[j]+=eps*sign
                yTest=xMat*wsTest
                rssE=rssError(yMat.A,yTest.A)
                if rssE<lowestError:
                    lowestError=rssE
                    wsMax=wsTest
        ws=wsMax.copy()
        returnMat[i,:]=ws.T
    return returnMat

xArr,yArr=loadDataSet('abalone.txt')
stageWise(xArr,yArr,0.01,200)



