from numpy import *

def loadDataSet(fileName,delim='\t'):
    fr=open(fileName)
    stringArr=[line.strip().split(delim) for line in fr.readlines() ]
    datArr=[list(map(float,line)) for line in stringArr]
    return mat(datArr)

# 2个参数：一个参数是用于进行PCA操作的数据集，第二个参数是可选参数，即应用N个特征
# 首先计算并减去原始数据集的平均值，然后计算协方差矩阵及其特征值
# 然后利用argsort函数对特征值进行从小到大排序
# 根据特征值排序的逆序就可以得到最大的N个向量
# 这些向量将构成后面对数据进行转换的矩阵
# 该矩阵则利用N个特征将原始数据转换到新空间中
# 最后原始数据被重构后返回
# 同时，降维之后的数据集也被返回
def pca(dataMat,topNfeat=9999999):
    meanVals=mean(dataMat,axis=0)
    meanRemoved=dataMat-meanVals # 去除平均值
    covMat=cov(meanRemoved,rowvar=0)
    eigVals,eigVects=linalg.eig(mat(covMat))
    eigValInd=argsort(eigVals)
    # 从大到小对N个值排序
    eigValInd=eigValInd[:-(topNfeat+1):-1]
    redEigVects=eigVects[:,eigValInd]
    # 将数据转换到新空间
    lowDDataMat=meanRemoved*redEigVects
    reconMat=(lowDDataMat*redEigVects.T)+meanVals
    return lowDDataMat,reconMat

dataMat=loadDataSet('testSet.txt')

lowDMat,reconMat=pca(dataMat,1)

import matplotlib.pyplot as plt

# fig=plt.figure()
# ax=fig.add_subplot(111)
# ax.scatter(dataMat[:,0].flatten().A[0],dataMat[:,1].flatten().A[0],marker='^',s=90)
# ax.scatter(reconMat[:,0].flatten().A[0],reconMat[:,1].flatten().A[0],marker='o',s=50,c='red')
# plt.show()

# 将NaN替换成平均值得函数
def replaceNanWithMean():
    datMat=loadDataSet('secom.data',' ')
    numFeat=shape(dataMat)[1]
    for i in range(numFeat):
        meanVal=mean(datMat[nonzero(~isnan(datMat[:,i].A))[0],i])
        # 计算所有非Nan的平均值
        # 将所有nan置为平均值
        datMat[nonzero(isnan(datMat[:,i].A))[0],i]=meanVal
    return datMat

dataMat=replaceNanWithMean()
