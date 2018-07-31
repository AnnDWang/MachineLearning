from numpy import *

def loadDataSet(fileName):
    dataMat=[]
    labelMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        lineArr=line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

# 在某个区间范围内随机选择一个整数
# i是第一个alpha的下标，m是所有alpha的数目
def selectJrand(i,m):
    j=i
    while(j==i):
        j=int(random.uniform(0,m))
    return j

# 用于在数值太大时对其进行调整
def clipAlpha(aj,H,L):
    if aj>H:
        aj=H
    if aj<L:
        aj=L
    return aj

dataArr,labelArr=sb=loadDataSet('testSet.txt')

# 简化版SMO算法
# 输入5个参数：分别是数据集、类别标签、常数C、容错率和退出前最大循环次数
# 将多个列表和输入参数转换成Numpy矩阵，这样就可以简化很多数学处理操作
# 转置了类别标签，因此我们得到的就是一个列向量而不是列表。
# 类别标签向量的每行元素和数据矩阵中的行一一对应
# 通过shaphe得到dataMatIn的m和n，最后，可以构建一个alpha列矩阵，矩阵中的元素都初始化为0
# 并建立一个iter变量，改变了存储的则是在没有任何alpha改变的情况下遍历数据集的次数
# 当改变了达到输入值maxIter时，函数结束运行并推出
# 每次循环当中，将alphaPairsChanged先设为为0，然后再对整个集合顺序遍历。
# 变量alphaPairsChanged用于记录alpha是否已经进行优化，
# 在循环结束时会得知这一点
# fXi能够计算出来，这就是我们所预测的类别，
# 然后基于这个实例的预测结果和真实结果比对，计算误差Ei
# 如果误差很大，对该数据实例所对应的alpha值进行优化
# 在if语句炸年糕，不管是正间隔还是负间隔都会被测试，并且在该if语句中，也要同时检查alpha值
# 以保证其不能等于0或者等于C
# 由于后面alpha小于0或大于C时将会被调整为0或C，所以一旦在该if语句中它们等于这两个值的话，那么它们就已经在边界了
# 因而不再能够减小或增大，因此就不值得再对它们进行优化了
# 接下来，利用辅助函数随机选择第二个alpha值，即alpha[j]
# 同样，可以采用第一个alpha值即alpha[i]的误差计算方法，来计算这个alpha值得误差。
# 这个过程可以通过copy方法来实现，因此稍后可以将新的alpha值与老的alpha值进行比较。
# python会通过引用的方式传递所有列表，所以必须明确告知python要为alphaIold和alphaJold分配新的内存
# 否则的话，对新值和旧值进行比较时，我们就看不到新旧值得变化。
# 之后开始计算L和H，用于将alpha[j]调整到0到C之间
# 如果L和H相等，就不做任何改变，直接执行continue语句
# eta是alpha[j]的最优修改量，在哪个很长的计算代码中得到。如果eta为0，那就是说需要退出for循环的当前迭代过程
# 如果eta为0，那么计算新的alpha[j]就比较麻烦。
# 需要检查alpha[j]是否有轻微改变
# 如果是，就退出for循环，然后alpha[i]和alpha[j]进行统一的改变，虽然改变的大小一样，丹是改变的方向正好相反
# 在对alpha[j]和alpha[i]进行优化之后，给这两个alpha值设置一个常数项b
# 在优化过程结束的同事，必须确保在合适的时机结束循环。
# 如果程序执行到for循环的最后一行都不执行continue语句，那么就已经成功改变了一堆alpha，同事可以增加alphaPairsChanged的值
# 在for循环之外，需要检查alpha值是否做了更新，如果有更新则将iter设为0之后继续运行程序
# 只有在所有数据集上遍历maxIter次，且不再发生任何alpha修改之后，程序次啊会停止并退出while循环
def smoSimple(dataMatIn,classLabels,C,toler,maxIter):
    dataMatrix=mat(dataMatIn)
    labelMat=mat(classLabels).transpose()
    b=0
    m,n=shape(dataMatrix)
    alphas=mat(zeros((m,1)))
    iter=0
    while(iter<maxIter):
        alphaPairsChanged=0
        for i in range(m):
            fXi=float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T))+b
            Ei=fXi-float(labelMat[i])
            # 如果alpha可以更改，进入优化过程
            if((labelMat[i]*Ei<-toler) and (alphas[i]<C)) or ((labelMat[i]*Ei>toler) and (alphas[i]>0)):
                j=selectJrand(i,m)# 随机选择第二个alpha
                fXj=float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T))+b
                Ej=fXj-float(labelMat[j])
                alphaIold=alphas[i].copy()
                alphaJold=alphas[j].copy()
                # 保证alpha在0与C之间
                if(labelMat[i]!=labelMat[j]):
                    L=max(0,alphas[j]-alphas[i])
                    H=min(C,C+alphas[j]-alphas[i])
                else:
                    L=max(0,alphas[j]+alphas[i]-C)
                    H=min(C,alphas[j]+alphas[i])
                if L==H:
                    print('L==H')
                    continue
                eta=2.0*dataMatrix[i,:]*dataMatrix[j,:].T-dataMatrix[i,:]*dataMatrix[i,:].T-dataMatrix[j,:]*dataMatrix[j,:].T
                if eta>-0:
                    print('eta>=0')
                    continue
                alphas[j]-=labelMat[j]*(Ei-Ej)/eta
                alphas[j]=clipAlpha(alphas[j],H,L)
                if(abs(alphas[j]-alphaJold)<0.00001):
                    print('j not moving enough')
                    continue
                # 对i进行修改，修改量与j相同，但方向相反
                alphas[i]+=labelMat[j]*labelMat[i]*(alphaJold-alphas[j])
                b1=b-Ei-labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T-labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2=b-Ej-labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T-labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if(0<alphas[i]) and (C>alphas[i]):b=b1
                elif(0<alphas[j]) and (C>alphas[j]):b=b2
                else:b=(b1+b2)/2.0
                alphaPairsChanged+=1
                print('iter: %d i: %d ,pairs changed %d' % (iter,i,alphaPairsChanged))
        if(alphaPairsChanged==0):iter+=1
        else: iter=0
        print('iteration number: %d' % iter)
    return b,alphas

b,alphas=smoSimple(dataArr,labelArr,0.6,0.001,40)

# 完整platt smo算法
# 选择alpha的方式不同
# 完整版的platt smo算法应用了一些能够提速的启发方法
# platt smo算法是通过一个外循环来选择第一个alpha值得，并且其选择过程会在两种方式之间进行交替
# 一种方式是在所有数据集上进行单边扫描，另一种方式则是在非边界alpha中实现单遍扫描
# 非边界alpha指的就是那些不等于边界0或者C的alpha值。
# 对整个数据集扫描相当容易，而实现非边界alpha值得扫描时，首先需要建立这些alpha值得列表，然后再对这个表进行遍历
# 同时会跳过那些已知的不会改变的alpha的值
# 在选择第一个alpha值后，算法会通过一个内循环来选择第二个alpha值。
# 在优化过程中，会通过最大化步长的方式获得第二个alpha值
# 在简化版smo中，会在选择j之后计算错误率Ej
# 在这里，会建立一个全局的缓存用于保存误差值，并从中选择使得步长或者说Ei-Ej最大的alpha值
class optStruct:
    def __init__(self,dataMatIn,classLabels,C,toler,kTup):
        self.X=dataMatIn
        self.labelMat=classLabels
        self.C=C
        self.tol=toler
        self.m=shape(dataMatIn)[0]
        self.alphas=mat(zeros((self.m,1)))
        self.b=0
        self.eCache=mat(zeros((self.m,2)))# 误差缓存
        # ktup是一个包含核函数信息的元组
        # 在初始化方法结束时，矩阵k先被构建，然后在通过调用函数kernerlTrans进行填充，全局的K值只需计算一次
        self.K=mat(zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i]=kernelTrans(self.X,self.X[i,:],kTup)

# 对于给定的alpha值，改函数能够计算E值并返回
def calcEk(oS,k):
    # fXk=float(multiply(oS.alphas,oS.labelMat).T*(oS.X*oS.X[k,:].T))+oS.b
    # Ek=fXk-float(oS.labelMat[k])
    # 使用核函数之后
    fXk=float(multiply(oS.alphas,oS.labelMat).T*oS.K[:,k]+oS.b)
    Ek=fXk-float(oS.labelMat[k])
    return Ek

# 用于选择第二个alpha或者说内循环的alpha值
# 目标是选择合适的第二个alpha值以保证每次优化中采用最大步长
# 该函数的误差值与第一个alpha值Ei和下标i有关，首先将输入值Ei在缓存中设置为有效的
# 有效意味着已经计算好了
# 在eCache中，代码nonzero(oS.eCache[:,0].A)[0]构建出一个非零表
# numpy函数nonzero返回了一个列表，而这个列表中包含以输入列表为目录的列表值
# 这里值为非0，nonzero语句返回的是非0E值对应的alpha值，而不是E值本身。
# 程序会在所有的值上进行循环并选择其中使得改变最大的那个值
# 如果是第一次循环，就随机选择一个alpha值
def selectJ(i,oS,Ei): # 内循环中的启发式方法
    maxK=-1
    maxDeltaE=0
    Ej=0
    oS.eCache[i]=[1,Ei]
    validEcacheList=nonzero(oS.eCache[:,0].A)[0]
    if(len(validEcacheList))>1:
        for k in validEcacheList:
            if k==i:
                continue
            Ek=calcEk(oS,k)
            deltaE=abs(Ei-Ek)
            if(deltaE>maxDeltaE):
                # 选择具有最大步长的j
                maxK=k
                maxDeltaE=deltaE
                Ej=Ek
        return maxK,Ej
    else:
        j=selectJrand(i,oS.m)
        Ej=calcEk(oS,j)
    return j,Ej

# 会计算误差值并存入缓存中，在alpha值优化之后会用到这个值
def updateEk(oS,k):
    Ek=calcEk(oS,k)
    oS.eCache[k]=[1,Ek]

# 优化例程
def innerL(i,oS):
    Ei=calcEk(oS,i)
    if((oS.labelMat[i]*Ei<-oS.tol) and (oS.alphas[i]<oS.C)) or ((oS.labelMat[i]*Ei>oS.tol) and (oS.alphas[i]>0)):
        # 第二个alpha选择中的启发式方法
        j,Ej=selectJ(i,oS,Ei)
        alphaIold=oS.alphas[i].copy()
        alphaJold=oS.alphas[j].copy()
        if(oS.labelMat[i]!=oS.labelMat[j]):
            L=max(0,oS.alphas[j]-oS.alphas[i])
            H=min(oS.C,oS.C+oS.alphas[j]-oS.alphas[i])
        else:
            L=max(0,oS.alphas[i]+oS.alphas[i]-oS.C)
            H=min(oS.C,oS.alphas[j]+oS.alphas[i])
        if L==H:
            print('L==H')
            return 0
        #eta=2.0*oS.X[i,:]*oS.X[j,:].T-oS.X[i,:]*oS.X[i,:].T-oS.X[j,:]*oS.X[j,:].T
        # 使用核函数时
        eta=2.0*oS.K[i,j]-oS.K[i,i]-oS.K[j,j]
        if eta>=0:
            print('eta>=0')
            return 0
        oS.alphas[j]-=oS.labelMat[j]*(Ei-Ej)/eta
        oS.alphas[j]=clipAlpha(oS.alphas[j],H,L)
        # 更新误差缓存
        updateEk(oS,j)
        if (abs(oS.alphas[j]-alphaJold)<0.00001):
            print('j not moving enough ')
            return 0
        oS.alphas[i]+=oS.labelMat[j]*oS.labelMat[i]*(alphaJold-oS.alphas[j])
        updateEk(oS,i) # 更新缓存
        #b1=oS.b-Ei-oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[i,:].T-oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[i,:]*oS.X[j,:].T
        #b2=oS.b-Ej-oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[j,:].T-oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[j,:]*oS.X[i,:].T
        # 使用核函数时
        b1=oS.b-Ei-oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i]-oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2=oS.b-Ei-oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]-oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        if(0<oS.alphas[i]) and (oS.C>oS.alphas[i]):
            oS.b=b1
        elif(0<oS.alphas[j]) and (oS.C>oS.alphas[j]):
            oS.b=b2
        else:
            oS.b=(b1+b2)/2.0
        return 1
    else:
        return 0

# 输入和smoSimpe完全一样，
# 整个代码的主体是while循环，这与smoSimple有些类似，但是这里的循环退出条件更多一些
# 当迭代次数超过指定的最大值，或者遍历整个集合都未对任意alpha对进行修改时，就退出循环
# 这里的maxIter遍历和函数smoSimple中的作用有一点不同，后者当没有任何alpha发生改变时会将整个集合的一次遍历过程计成一次迭代
# 而这里的依次迭代定义为一次循环过程
# 而不管该循环具体做了什么事，如果在优化过程中存在波动就会停止
# 这里的做法优于smoSimple函数中的计数方法
# while循环内部与smoSimple中有所不同，一开始的for循环在数据集上遍历任意可能的alpha
# 通过调用innerL来选择第二个alpha
# 并在可能时对其进行优化处理
# 如果有任意一对alpha值发生改变，那么会返回1
# 第二个for循环遍历所有的非边界alpha值，也就是不在边界0或C上的值

def smoP(dataMatIn,classLabels,C,toler,maxIter,kTup=('lin',0)):
    oS=optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler,kTup)
    iter=0
    entireSet=True
    alphaPairsChanged=0
    while(iter<maxIter) and ((alphaPairsChanged>0) or (entireSet)):
        alphaPairsChanged=0
        # 遍历所有的值
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged+=innerL(i,oS)
            print('fullset, iter: %d i: %d,  pairs changed %d'%(iter,i,alphaPairsChanged))
            iter+=1
        else:# 遍历非边界值
            nonBoundIs=nonzero((oS.alphas.A>0)*(oS.alphas.A<C))[0]
            for i in nonBoundIs:
                alphaPairsChanged+=innerL(i,oS)
                print('non-bound, iter: %d i: %d, pairs changed %d' %(iter,i,alphaPairsChanged))
            iter+=1
        if entireSet: entireSet=False
        elif (alphaPairsChanged==0):
            entireSet=True

        print('iteration number: %d ' %iter)
    return oS.b,oS.alphas

#b,alphas=smoP(dataArr,labelArr,0.6,0.001,40)

#
def calcWs(alphas,dataArr,classLabels):
    X=mat(dataArr)
    labelMat=mat(classLabels).transpose()
    m,n=shape(X)
    w=zeros((n,1))
    for i in range(m):
        w+=multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w

# 核转换函数
# 该函数有3个输入参数：2个数值型变量和1个元组
# 元组kTup给出的是核函数的信息
# 元组的第一个参数是描述所用核函数类型的一个字符串
# 其它两个参数则都是核函数可能的可选参数
# 该函数首先构建出一个列向量
# 然后检查元组以确定核函数的类型
# 在线性核函数的情况下，内积计算在所有数据集合数据集中的一行这两个输入之间展开，
# 在径向基核函数的情况下，在for循环中对于矩阵的每个元素计算搞死函数的值
# 在for循环结束之后，我们将计算过程应用到整个向量上去。
# 在numpy矩阵中，除法符号意味着对矩阵元素展开计算而不像在matlab中一样计算矩阵的逆
def kernelTrans(X,A,kTup):
    m,n=shape(X)
    K=mat(zeros((m,1)))
    if kTup[0]=='lin':
        K=X*A.T
    elif kTup[0]=='rbf':
        for j in range(m):
            deltaRow=X[j,:]-A
            K[j]=deltaRow*deltaRow.T
        K=exp(K/(-1*kTup[1]**2))# 元素之间的除法
    else:
        raise NameError('Houston we have a problem -- the kernel is not recognized')
    return K

ws=calcWs(alphas,dataArr,labelArr)

# 对第一个数据点进行分类
dataMat=mat(dataArr)
dataMat[0]*mat(ws)+b

# 利用核函数进行分类的径向基测试函数
def testRbf(k1=1.3):
    dataArr,labelArr=loadDataSet('testSetRBF.txt')
    b,alphas=smoP(dataArr,labelArr,200,0.0001,10000,('rbf',k1))
    dataMat=mat(dataArr)
    labelMat=mat(labelArr).transpose()
    svInd=nonzero(alphas.A>0)[0]
    sVs=dataMat[svInd] # 构建支持向量矩阵
    labeSV=labelMat[svInd]
    print('there are %d support vectors '%shape(sVs)[0])
    m,n=shape(dataMat)
    errorCount=0
    for i in range(m):
        kernelEval=kernelTrans(sVs,dataMat[i,:],('rbf',k1))
        predict=kernelEval.T*multiply(labeSV,alphas[svInd])+b
        if sign(predict)!=sign(labelArr[i]): errorCount+=1
    print('the training error rate is: %f' %(float(errorCount)/m))
    dataArr,labelArr=loadDataSet('testSetRBF2.txt')
    errorCount=0
    dataMat=mat(dataArr)
    labelMat=mat(labelArr).transpose()
    m,n=shape(dataMat)
    for i in range(m):
        kernelEval=kernelTrans(sVs,dataMat[i,:],('rbf',k1))
        predict=kernelEval.T*multiply(labeSV,alphas[svInd])+b
        if sign(predict)!=sign(labelArr[i]): errorCount+=1
    print('the test error rate is : %f' %(float(errorCount)/m))

testRbf()

# 支持向量的数目存在一个最优值，SVM的有点在于它能对数据进行高效分了。
# 如果支持向量太少，就可能会得到一个很差的决策边界
# 如果支持向量太多，也就相当于每次都利用整个数据集进行分类





















