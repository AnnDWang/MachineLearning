from numpy import *

def loadExtData():
    return [[1,1,1,0,0],
            [2,2,2,0,0],
            [1,1,1,0,0],
            [5,5,5,0,0],
            [1,1,0,2,2],
            [0,0,0,3,3],
            [0,0,0,1,1]]

Data=loadExtData()
U,Sigma,VT=linalg.svd(Data)

Sig3=mat([[Sigma[0],0,0],
          [0,Sigma[1],0],
          [0,0,Sigma[2]]])

from numpy import linalg as la

# 相似度计算
def ecludSim(inA,inB):
    return 1.0/(1.0+la.norm(inA-inB))

# 会检查是否存在3个或更多的点，如果不存在，函数返回1
def pearsSim(inA,inB):
    if len(inA)<3:return 1.0
    return 0.5+0.5*corrcoef(inA,inB,rowvar=0)[0][1]

def cosSim(inA,inB):
    num=float(inA.T*inB)
    denom=la.norm(inA)*la.norm(inB)
    return 0.5+0.5*(num/denom)

# 基于物品相似度的推荐引擎
def standEst(dataMat,user,simMeas,item):
    n=shape(dataMat)[1]
    simTotal=0.0
    ratSimTotal=0.0
    for j in range(n):
        userRating=dataMat[user,j]
        if userRating==0:continue
        # 寻找两个用户都评级的物品
        overLap=nonzero(logical_and(dataMat[:,item].A>0,dataMat[:,j].A>0))[0]
        if len(overLap)==0:
            similarity=0
        else:
            similarity=simMeas(dataMat[overLap,item],dataMat[overLap,j])
        print('the %d and %d similarity is : %f' %(item,j,similarity))
        simTotal+=similarity
        ratSimTotal+=similarity*userRating
    if simTotal==0:return 0
    else:
        return ratSimTotal/simTotal

def recommend(dataMat,user,N=3,simMeas=cosSim,estMethod=standEst):
    # 寻找未评级的物品
    unratedItems=nonzero(dataMat[user,:].A==0)[1]
    if len(unratedItems)==0: return 'you rated everything'
    itemScores=[]
    for item in unratedItems:
        estimatedScore=estMethod(dataMat,user,simMeas,item)
        itemScores.append((item,estimatedScore))
    # 返回前N个未评级物品
    return sorted(itemScores,key=lambda jj:jj[1],reverse=True)[:N]

def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]

# 基于SVD的评分估计
def svdEst(dataMat,user,simMeas,item):
    n=shape(dataMat)[1]
    simTotal=0.0
    ratSimTotal=0.0
    U,Sigma,VT=la.svd(dataMat)#建立对角矩阵
    Sig4=mat(eye(4)*Sigma[:4])
    xformedItems=dataMat.T*U[:,:4]*Sig4.I
    # 构建转换后的物品
    for j in range(n):
        userRating=dataMat[user,j]
        if userRating==0 or j==item:
            continue
        similarity=simMeas(xformedItems[item,:].T,xformedItems[j,:].T)
        print('the %d and %d similarity is : %f' %(item,j,similarity))
        simTotal+=similarity
        ratSimTotal+=similarity*userRating
    if simTotal==0:return 0
    else:
        return ratSimTotal/simTotal
# myMat=mat(loadExData2())
# recommend(myMat,1,estMethod=svdEst)

# 图像压缩函数
def printMat(inMat,thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(inMat[i,k])>thresh:
                print(1)
            else:
                print(0)
        print(' ')

def imgCompress(numSV=3,thresh=0.8):
    myl=[]
    for line in open('0_5.txt').readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = mat(myl)
    print("****original matrix******")
    printMat(myMat, thresh)
    U, Sigma, VT = la.svd(myMat)
    SigRecon = mat(zeros((numSV, numSV)))
    for k in range(numSV):  # construct diagonal matrix from vector
        SigRecon[k, k] = Sigma[k]
    reconMat = U[:, :numSV] * SigRecon * VT[:numSV, :]
    print("****reconstructed matrix using %d singular values******" % numSV)
    printMat(reconMat, thresh)
    # for line in open('0_5.txt').readlines():
    #     newRow=[]
    #     for i in range(32):
    #         newRow.append(int(line[i]))
    #     myl.append(newRow)
    # myMat=mat(myl)
    # print("************original matrix****************")
    # print(myMat,thresh)
    # U,Sigma,VT=la.svd(myMat)
    # SigRecon=mat(zeros((numSV,numSV)))
    # for k in range(numSV):
    #     SigRecon[k,k]=Sigma[k]
    # reconMat=U[:,:numSV]*SigRecon*VT[:numSV,:]
    # print("****reconstructed matrix using %d singular values**********************" % numSV)
    # printMat(reconMat,thresh)

imgCompress(2)