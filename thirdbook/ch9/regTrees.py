from numpy import *

class treeNode():
    def __init__(self,feat,val,right,left):
        featureToSplitOn=feat
        valueOfSplit=val
        rightBranch=right
        leftBranch=left

# CART 算法的实现代码
def loadDataSet(fileName):
    dataMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        curLine=line.strip().split('\t')
        fltLine=list(map(float,curLine))# 将每行映射成浮点数
        dataMat.append(fltLine)
    return dataMat

# 数据集合，待切分的特征和该特征的某个值
def binSplitDataSet(dataSet,feature,value):
    # mat0=dataSet[nonzero(dataSet[:,feature]>value)[0],:][0]
    # mat1=dataSet[nonzero(dataSet[:,feature]<=value)[0],:][0]
    mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0,mat1

# 负责生成叶节点，当chooseBestSplit函数确定不再对数据进行切分时，将调用该regLeaf函数来得到叶节点模型
def regLeaf(dataSet):#returns the value used for each leaf
    return mean(dataSet[:,-1])

# 误差估计函数regErr，该函数在给定数据上计算目标变量的平方误差
def regErr(dataSet):
    return var(dataSet[:,-1]) * shape(dataSet)[0]

# 是回归树构建的核心函数，该函数的目的是找到数据的最佳二元切分方式，如果找不到好的二元切分
# 该函数返回None并同时调用CreateTree方法来产生叶节点
# 叶节点的值也将返回None
# 一开始为ops制定了tols和tolN两个值
# 它们是用户指定的参数，用于控制函数的停止时机
# 其中变量tols是容许的误差下降值
# tolN是切分的最少样本数
# 接下来通过对当前所有目标变量建立一个集合
# 函数会统计不同剩余特征值得数目
# 如果该数目为1，那么就不需要再切分而且直接返回
# 然后函数计算了当前数据集的大小和误差
# 该误差S将用于新切分误差的进行对比，来检查新切分能够降低误差
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    tolS = ops[0]; tolN = ops[1]
    #if all the target variables are the same value: quit and return value
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1: #exit cond 1
        return None, leafType(dataSet)
    m,n = shape(dataSet)
    #the choice of the best feature is driven by Reduction in RSS error from mean
    S = errType(dataSet)
    bestS = inf; bestIndex = 0; bestValue = 0
    for featIndex in range(n-1):
        # for splitVal in set(dataSet[:,featIndex]):
        for splitVal in set(dataSet[:, featIndex].T.A.tolist()[0]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    #if the decrease (S-bestS) is less than a threshold don't do the split
    # 如果误差减少不大则退出
    if (S - bestS) < tolS:
        return None, leafType(dataSet) #exit cond 2
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    # 如果切分的数据集很小则退出
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):  #exit cond 3
        return None, leafType(dataSet)
    return bestIndex,bestValue#returns the best feature to split on
                              #and the value used for that split


# 四个参数：数据集合其他三个可选参数
# leafType给出建立叶节点的函数
# errType代表误差计算函数
# ops是一个包含树构建所需其他参数的元组
# 该函数首先尝试将数据集分成两个部分，满足停止条件，则返回None和某类型的值
# 如果构建回归树，则该模型是一个常数，如果是模型树，其模型是一个线性方程
def createTree(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    feat,val=chooseBestSplit(dataSet,leafType,errType,ops)
    # 满足停止条件时返回叶节点
    if feat==None: return val
    retTree={}
    retTree['spInd']=feat
    retTree['spVal']=val
    lSet,rSet=binSplitDataSet(dataSet,feat,val)
    retTree['left']=createTree(lSet,leafType,errType,ops)
    retTree['right']=createTree(rSet,leafType,errType,ops)
    return retTree

myDat=loadDataSet('ex00.txt')
myMat=mat(myDat)
createTree(myMat)

myDat1=loadDataSet('ex0.txt')
myMat1=mat(myDat1)
createTree(myMat1)

# regTreeEval 和 modelTreeEval对回归树节点预测
# 它们会对输入数据进行格式化处理
# 在原数据矩阵上增加第0列，然后计算并返回预测值
# 为了与函数modelTreeEval保持一致，尽管regTreeEval只使用一个输入，但仍然保留了两个输入参数

# 对回归树叶节点进行预测
def regTreeEval(model,inDat):
    return float(model)

# 对模型树节点进行预测
def modelTreeEval(model,inDat):
    n=shape(inDat)[1]
    X=mat(ones((1,n+1)))
    X[:,1:n+1]=inDat
    return float(X*model)

def isTree(obj):
    return (type(obj).__name__=='dict')

# 是一个递归函数，从上往下遍历树直到叶节点为止
# 如果找到两个叶节点则计算他们的平均值
# 该函数对树进行塌陷处理（即返回树平均值）
# 在prune函数中调用该函数时应明确说明这一点
def getMean(tree):
    if isTree(tree['right']):tree['right']=getMean(tree['right'])
    if isTree(tree['left']):tree['left']=getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0

# 两个参数，待剪枝的树与剪枝所需的测试数据
# 首先确认测试集是否为空，一旦非空，反复递归调用函数对测试数据进行切分
# 因为树是由其他数据集生成的，所以测试集上会有一些样本与原始数据集样本的取值范围不同
# 一旦出现这种情况，假设发生了过拟合，对树进行剪枝
# 如果是子树，那么就继续剪枝
# 对左右两个子树完成剪枝之后，还需要继续检查他们仍然还是子树，如果两个分支已经不再是子树
# 那么就进行合并，对合并前后误差进行对比，确定是否进行合并操作
def prune(tree,testData):
    if shape(testData)[0]==0: return getMean(tree)
    # 没有测试数据对树进行塌陷处理
    if (isTree(tree['right']) or isTree(tree['left'])):
        lSet,rSet=binSplitDataSet(testData,tree['spInd'],tree['spVal'])
    if isTree(tree['left']):tree['left']=prune(tree['left'],lSet)
    if isTree(tree['right']):tree['right']=prune(tree['right'],rSet)
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet,rSet=binSplitDataSet(testData,tree['spInd'],tree['spVal'])
        errorNoMerge=sum(power(lSet[:,-1]-tree['left'],2))+sum(power(rSet[:,-1]-tree['right'],2))
        treeMean=(tree['left']+tree['right'])/2.0
        errorMerge=sum(power(testData[:,-1]-treeMean,2))
        if errorMerge<errorNoMerge:
            print('merging')
            return treeMean
        else:
            return tree
    else:
        return tree


# 对于输入的单个数据点或者行向量，函数treeForeCast会返回一个浮点值
# 在给定树结构的情况下，对于单个数据点，该函数会给出一个预测值
# 调用函数treeForeCast时需要指定树的类型
# 以便在叶节点上能够调用合适的模型
# 参数modelEval是对叶节点数据进行预测的函数的引用
# 函数treeForeCast自顶向下遍历整颗树，知道命中叶节点为止
# 一旦到达叶节点，就会在输入数据上调用modelEval函数，而函数的默认值是regTreeEval
def treeForeCast(tree,inData,modelEval=regTreeEval):
    if not isTree(tree): return modelEval(tree,inData)
    if inData[tree['spInd']]>tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'],inData,modelEval)
        else:
            return modelEval(tree['left'],inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'],inData,modelEval)
        else:
            return modelEval(tree['right'],inData)

# 多次调用treeForeCast函数，能够以向量的形式返回一组预测值，因此该函数在对整个测试集进行预测时非常有用
def createForeCast(tree,testData,modelEval=regTreeEval):
    m=len(testData)
    yHat=mat(zeros((m,1)))
    for i in range(m):
        yHat[i,0]=treeForeCast(tree,mat(testData[i]),modelEval)

myDat2=loadDataSet('ex2.txt')
myMat2=mat(myDat2)
myTree=createTree(myMat2,ops=(0,1))

myDatTest=loadDataSet('ex2test.txt')
myMat2Test=mat(myDatTest)
print(prune(myTree,myMat2Test))

# 模型树的叶节点生成函数
# 主要功能是将数据集格式化成目标变量Y和自变量X
# X和Y用于执行简单的线性回归
#
def linearSolve(dataSet):
    m,n=shape(dataSet)
    # 将X与Y中的数据格式化
    X=mat(ones((m,n)))
    Y=mat(ones((m,1)))
    X[:,1:n]=dataSet[:,0:n-1]
    Y=dataSet[:,-1]
    xTx=X.T*X
    if linalg.det(xTx)==0.0:
        raise NameError('this matrix is singuar, cannot do inverse, \n try increasing the second value of ops')
    ws=xTx.I*(X.T*Y)
    return ws,X,Y

# 当数据不再需要切分的时候，负责生成叶节点的模型

def modelLeaf(dataSet):
    ws,X,Y=linearSolve(dataSet)
    return ws

# 在给定数据集上计算误差
def modelErr(dataSet):
    ws,X,Y=linearSolve(dataSet)
    yHat=X*ws
    return sum(power(Y-yHat,2))

myMat2=mat(loadDataSet('exp2.txt'))

print(createTree(myMat2,modelLeaf,modelErr,(1,10)))





