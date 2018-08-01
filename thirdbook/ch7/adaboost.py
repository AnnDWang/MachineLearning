from numpy import *

def loadSimpData():
    datMat=matrix([[1.,2.1],
                   [2.,1.1],
                   [1.3,1.],
                   [1.,1.],
                   [2.,1.]])
    classLabels=[1.0,1.0,-1.0,-1.0,1.0]
    return datMat,classLabels

# 单层决策树生成树

# 通过阈值比较对数据进行分类的。
# 所有在阈值一边的数据会分类到类别-1，而在另外一边的数据分到类别+1
# 该函数可以通过数组过滤来实现
# 首先将返回数组的全部元素设置为1
# 然后将所有不满足不等式要求的元素设置为-1
# 可以基于数据集中的任一元素进行比较，同时也可以将不等号在大于小于之间切换
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    retArray=ones((shape(dataMatrix)[0],1))
    if threshIneq=='lt':
        retArray[dataMatrix[:,dimen]<=threshVal]=-1.0
    else:
        retArray[dataMatrix[:,dimen]>threshVal]=-1.0

    return retArray

# 第二个函数buildStump将会遍历stumpClassify函数所有的可能输入值
# 并找到数据集上最佳的单层决策树
# 最佳是基于数据的权重向量D来定义的
# 在确保输入数据符合矩阵格式之后，整个函数就开始执行了
# 函数将构建一个称为bestStump的空字典，这个字典用于存储给定权重向量D时所得到的最佳单层决策树的相关信息
# 变量numSteps用于在特征的所有可能值上进行遍历
# 而遍历minError则在一开始就初始化成正无穷大，之后用于寻找可能的最小错误率
# 三层嵌套的for循环是程序最主要的部分。
# 第一层for循环在数据集的所有特征上遍历
# 考虑数值型的特征，我们就可以通过计算最小值和最大值来了解应该需要多大的步长
# 第二层for循环再在这些值上遍历。甚至将阈值设置为整个取值范围之外也是可以的。
# 因此在取值范围之外应该还有两个额外步骤
# 最后一个for循环则是在大于和小于之间切换不等式
# 在嵌套的三层for循环之内，我们在数据集及三个循环变量上调用stumpClassify函数
# 基于这些循环变量，该函数会返回分类预测结果
# 接下来构建一个列向量errArr，如果predcitVals中的值不等于labelMat中的真正类别标签值
# 那么errArr的相应位置为1
# 将错误向量errArr和权重向量D的相应元素相乘并求和，就得到了数值weightedError
# 这就是AdaBoost和分类器交互的地方
# 这里，我们是基于权重向量D而不是其他错误计算指标来评价分类器的
# 如果需要使用其他分类器的话，就需要考虑D上最佳分类器所定义的计算过程
# 最后，将当前的错误率与已有的最小错误率进行对比
# 如果当前的值较小，那么就在词典bestStump中保存该单层决策树
# 字典、错误率和类别估计值都会返回AdaBoost算法
def buildStump(dataArr,classLabels,D):
    dataMatrix=mat(dataArr)
    labelMat=mat(classLabels).T
    m,n=shape(dataMatrix)
    numSteps=10.0
    bestStump={}
    bestClassEst=mat(zeros((m,1)))
    minError=inf
    for i in range(n):
        rangeMin=dataMatrix[:,i].min()
        rangeMax=dataMatrix[:,i].max()
        stepSize=(rangeMax-rangeMin)/numSteps
        for j in range(-1,int(numSteps)+1):
            for inequal in ['lt','gt']:
                threshVal=(rangeMin+float(j)*stepSize)
                predictVals=stumpClassify(dataMatrix,i,threshVal,inequal)
                errArr=mat(ones((m,1)))
                errArr[predictVals==labelMat]=0
                weightedError=D.T*errArr # 计算加权错误率
                print('split: dim %d,htresh %.2f, thresh inequal: %s, the weighted error is %.3f' % (i,threshVal,inequal,weightedError))
                if weightedError<minError:
                    minError=weightedError
                    bestClassEst=predictVals.copy()
                    bestStump['dim']=i
                    bestStump['thresh']=threshVal
                    bestStump['ineq']=inequal
    return bestStump,minError,bestClassEst

D=mat(ones((5,1))/5)
datMat,classLabels=loadSimpData()
#buildStump(datMat,classLabels,D)

# 基于单层决策树的AdaBoost训练过程
# 这个算法会输出一个单层决策树的数组，因此首先需要建立一个新的python表来对其进行存储
# 然后，得到数据集中的数据点的数目m，并建立一个列向量D
# 向量D非常重要，包含了每个数据点的权重
# 一开始，这些权重都赋予了相等的值
# 在后续的迭代中，adaboost算法会在增加错分数据权重的同事，降低正确分类数据的权重
# D是一个概率分布向量，因此其所有元素之和为1.0
# 一开始的所有元素都会被初始化为1/m
# 同时，程序还会建立另一个列向量aggClassEst，
# 记录每个数据点的类别估计累计值
# AdaBoost算法的核心在于for循环，该循环运行numIt次或者直到训练错误率为0为止
# 循环中的第一件事就是利用前面介绍的buildStump函数建立一个单层决策树
# 该函数的输入为权重向量D，返回的则是利用D而得到的具有最小错误率的单层决策树
# 同时返回的还有最小的错误率以及估计的类别向量
# 接下来，需要计算alpha的值，该值会告诉分类器本次单层决策树输出结果的权重
# 其中的max(error,1e-16)用于确保在没有错误时不会发生除0溢出
# 而后，alpha值加入到bestStump字典中，该字典又添加到列表中
# 该字典包括了分类所需要的所有信息
# 接下来，计算下一次迭代中的新权重向量D，在训练错误率为0时，就要提前结束for循环
# 此程序是通过aggClassEst变量保持一个运行时的类别估计值来实现的
# 该值只是一个浮点数，为了得到二值分类结果还需呀调用sign()函数
# 如果总错误率为0，则有break语句终止for循环
def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    weakClassArr=[]
    m=shape(dataArr)[0]
    D=mat(ones((m,1))/m)
    aggClassEst=mat(zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst=buildStump(dataArr,classLabels,D)
        print("D: ",D.T)
        alpha=float(0.5*log((1.0-error)/max(error,1e-16)))
        bestStump['alpha']=alpha
        weakClassArr.append(bestStump)
        print('classEst: ',classEst.T)
        # 为下一次迭代计算D
        expon=multiply(-1*alpha*mat(classLabels).T,classEst)
        D=multiply(D,exp(expon))
        D=D/D.sum()
        aggClassEst+=alpha*classEst
        print('aggClassEst: ',aggClassEst.T)
        # 错误率累加计算
        aggErrors=multiply(sign(aggClassEst)!=mat(classLabels).T,ones((m,1)))
        errorRate=aggErrors.sum()/m
        print('total error: ',errorRate,'\n')
        if errorRate==0.0:
            break
    return weakClassArr,aggClassEst

classifierArray=adaBoostTrainDS(datMat,classLabels,9)
#print(classifierArray)

# AdaBoost分类函数
# 利用训练出的多个弱分类器进行分类
# 该函数的输入是由一个或者多个待分类样例datToClass以及多个弱分类器组成的数组classifierArr
# 函数首先将datToClass转换成了一个numpy矩阵，并且得到其中的待分类样例个数m
# 然后构建一个0向量aggClassEst
# 这个列向量与adaBoostTrainDS中的含义一样
# 接下来，遍历classifierArr中的所有弱分类器
# 并给予stumpClassify对每个分类器得到一个类别的估计值
# 在前面构建单层决策树是，已经见过了stumpClassify函数
# 在那里，在所有可能的树桩值上进行迭代得到具有最小加权错误率的单层决策树
# 这里只是简单地应用了单层决策树
# 输出的类别估计值乘上该单层决策树的alpha权重然后累加到aggClassEst上，就完成了这一过程
def adaClassify(datToClass,classifierArr):
    dataMatrix=mat(datToClass)
    m=shape(dataMatrix)[0]
    aggClassEst=mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst=stumpClassify(dataMatrix,classifierArr[i]['dim'],classifierArr[i]['thresh'],classifierArr[i]['ineq'])
        aggClassEst+=classifierArr[i]['alpha']*classEst
        print(aggClassEst)
    return  sign(aggClassEst)
# datArr,labelArr=loadSimpData()
# classifierArr=adaBoostTrainDS(datArr,labelArr,30)
# adaClassify([0,0],classifierArr)
# print()

# 自适应数据加载
def loadDataSet(fileName):
    numFeat=len(open(fileName).readline().split('\t'))
    dataMat=[]
    labelMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        lineArr=[]
        curLine=line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

# datArr,labelArr=loadDataSet('horseColicTraining2.txt')
# classifierArray,aggClassEst=adaBoostTrainDS(datArr,labelArr,10)
#
# testArr,testLabelArr=loadDataSet('horseColicTest2.txt')
# prediction10=adaClassify(testArr,classifierArray)
#
# errArr=mat(ones((67,1)))
# errArr[prediction10!=mat(testArr).T].sum()
#
# print(errArr[prediction10!=mat(testLabelArr).T].sum())

import matplotlib.pyplot as plt
# ROC曲线的绘制以及AUC计算函数
def plotROC(predStrengths,classLabels):
    cur=(1.0,1.0)
    ySum=0.0
    numPosClas=sum(array(classLabels)==1.0)
    yStep=1/float(numPosClas)
    xStep=1/float(len(classLabels)-numPosClas)
    # 获取排好序的索引
    sortedIndicies=predStrengths.argsort()
    fig=plt.figure()
    fig.clf()
    ax=plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index]==1.0:
            delX=0
            delY=yStep
        else:
            delX=xStep
            delY=0
            ySum+=cur[1]
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY],c='b')
        cur=(cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    ax.axis([0,1,0,1])
    plt.show()
    print("the area under the cure is :",ySum*xStep)

datArr,labelArr=loadDataSet('horseColicTraining2.txt')
classifierArray,aggClassEst=adaBoostTrainDS(datArr,labelArr,10)
plotROC(aggClassEst.T,labelArr)