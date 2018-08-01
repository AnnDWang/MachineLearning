from numpy import *

def loadDataSet(fileName):
    dataMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        curLine=line.strip().split('\t')
        fltLine=list(map(float,curLine))
        dataMat.append(fltLine)
    return  dataMat

# 计算两个向量的欧式距离
def distEclud(vecA,vecB):
    return sqrt(sum(power(vecA-vecB,2)))

def randCent(dataSet,k):
    n=shape(dataSet)[1]
    centroids=mat(zeros((k,n)))
    # 构建簇质心
    for j in range(n):
        minJ=min(dataSet[:,j])
        rangeJ=float(max(dataSet[:,j])-minJ)
        centroids[:,j]=minJ+rangeJ*random.rand(k,1)
    return centroids

# k-均值聚类算法
# 4个输入参数，只有数据集合簇的数目是必须的
# 用来计算距离和创建初始质心的函数都是可选的
# 一开始确定数据集中数据点的总数，然后创建一个矩阵来存储每个点的分配结果。
# 簇分配结果矩阵clusterAssment包含两列，一列记录索引值，第二列存储误差
# 这里的误差是指当前点到簇质心的举例
# 后面会用该误差来评价聚类的效果
#
def kMeans(dataSet,k,distMeas=distEclud,createCent=randCent):
    m=shape(dataSet)[0]
    clusterAssment=mat(zeros((m,2)))
    centroids=createCent(dataSet,k)
    clusterChanged=True
    while clusterChanged:
        clusterChanged=False
        for i in range(m):
            minDist=inf
            minIndex=-1
            for j in range(k):
                distJI=distMeas(centroids[j,:],dataSet[i,:])
                if distJI<minDist:
                    minDist=distJI
                    minIndex=j
            if clusterAssment[i,0]!=minIndex:clusterChanged=True
            clusterAssment[i,:]=minIndex
            minDist**2
        print(centroids)
        for cent in range(k):
            ptsInClust=dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]
            centroids[cent,:]=mean(ptsInClust,axis=0)
    return centroids,clusterAssment

datMat=mat(loadDataSet('testSet.txt'))
myCentroids,clustAssing=kMeans(datMat,4)

# 二分K-均值聚类算法
# 该函数首先创建一个矩阵来存储数据集中每个点的簇分配结果及平方误差，然后计算整个数据集的质心
# 并使用一个列表来保留所有的质心。得到上述质心之后，可以遍历数据集中所有点来计算每个点到质心的误差值
# 接下来进行while循环，该循环不停地对簇进行划分，直到得到想要的簇数目位置
# 可以通过考察簇列表中的值来获得当前簇的数目
# 然后遍历所有的簇来决定最佳的簇进行划分
# 为此需要比较划分前后的SSE，一开始将SSE设置为无穷大
# 然后遍历簇列表centList中的每一个簇
# 对每一个簇，将该簇中的所有点看成一个小的数据集ptsInCurrCluster
# 将ptsInCurrCluster输入到kMeans中进行处理，k均值算法会生成两个质心，同时给出每个簇的误差值
# 这些误差与剩余数据集的误差之和作为本次划分的误差
# 如果该划分的SSE值最小，则本次划分被保存
# 一旦决定了要划分的簇，接下来就要实际执行划分操作
# 划分操作很容易，只需要将要划分的簇中所有点的簇分配结果进行修改即可
# 当使用kMeans函数并且指定k=2时，会得到两个编号分别为1和0的结果簇
# 需要将这些簇编号修改为划分簇及新加簇的编号
# 该过程可以通过两个数组过滤器来完成，最后，新的簇分配结果被更新，新的质心被添加到centList中
def biKmeans(dataSet,k,distMeas=distEclud):
    m=shape(dataSet)[0]
    clusterAssment=mat(zeros((m,2)))
    centroid0=mean(dataSet,axis=0).tolist()[0]
    centList=[centroid0]
    for j in range(m):
        clusterAssment[j,1]=distMeas(mat(centroid0),dataSet[j,:])**2
    while (len(centList)<k):
        lowestSSE=inf
        for i in range(len(centList)):
            ptsInCurrCluster=dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]
            centroidMat,splitClustAss=kMeans(ptsInCurrCluster,2,distMeas)
            sseSplit=sum(splitClustAss[:,1])
            sseNotSplit=sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
            print('sseSplit, and notSplit: ',sseSplit,sseNotSplit)
            if(sseSplit+sseNotSplit)<lowestSSE:
                bestCentToSplit=i
                bestNewCents=centroidMat
                bestClustAss=splitClustAss.copy()
                lowestSSE=sseSplit+sseNotSplit
        # 更新簇分配的结果
        bestClustAss[nonzero(bestClustAss[:,0].A==1)[0],0]=len(centList)
        bestClustAss[nonzero(bestClustAss[:,0].A==0)[0],0]=bestCentToSplit
        print('the bestCentToSplit is : ', bestCentToSplit)
        print('the len of bestClusAss is: ',len(bestClustAss))
        centList[bestCentToSplit]=bestNewCents[0,:].tolist()[0]
        centList.append(bestNewCents[1,:].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:,0].A==bestCentToSplit)[0],:]=bestClustAss
    return mat(centList),clusterAssment


datMat3=mat(loadDataSet('testSet2.txt'))
centList,myNewAssments=biKmeans(datMat3,3)
