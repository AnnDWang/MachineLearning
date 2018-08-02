from numpy import *

# apriori算法中的辅助函数
def loadDataSet():
    return [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]

# 将构建集合C1，C1是大小为1的所有候选项集的集合
# 算法先构建集合C1，然后扫描数据集来判断这些只有一个元素的项集是否满足最小支持度的要求
# 哪些满足最低要求的项集构成集合L1，L1中的元素相互组合构成C2,C2再进一步过滤变成L2
def createC1(dataSet):
    C1=[]
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    # frozenset 不可变集合，之后会将这些集合作为字典键值使用
    return list(map(frozenset,C1))

# 三个参数，分别是数据集、候选项集列表Ck以及感兴趣项集的最小支持度minSupport
# 该函数用于从C1生成L1。该函数会返回一个包含支持度值得字典以备后用
# scanD函数首先创建一个空字典ssCnt，然后遍历数据集中的所有交易记录以及C1中的候选集
# 如果C1中的集合是记录的一部分，那么增加字典中对应的计数值
# 这里字典的键就是集合
# 当扫描完数据集中的所有项以及所有候选集时，就需要计算支持度
# 不满足最小支持度的集合不会输出
# 函数会先构建一个空列表，该列表包含满足最小支持度要求的集合
# 下一个循环遍历字典中的每个元素并且计算支持度
# 如果支持度满足最小支持度的要求，就将该元素添加到retList中，
# 可以使用语句retList.insert(0,key)在列表的首部插入任意新的集合
# 函数最后返回的最频繁项集的支持度supportData
def scanD(D,Ck,minSupport):
    ssCnt={}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if can not in ssCnt:ssCnt[can]=1
                else:
                    ssCnt[can]+=1
    numItems=float(len(D))
    retList=[]
    supportData={}
    for key in ssCnt:
        support=ssCnt[key]/numItems# 计算所有项集的支持度
        if support>=minSupport:
            retList.insert(0,key)
        supportData[key]=support
    return retList,supportData

dataSet=loadDataSet()

C1=createC1(dataSet)

print(C1)

D=list(map(set,dataSet))

L1,supportData0=scanD(D,C1,0.5)

# apriori算法

# 输入参数为频繁项集列表与项集元素个数k，输出为Ck
# 首先创建一个空列表，计算Lk中的元素数目
# 接下来，比较Lk中的每一个元素与其他元素
# 可以通过两个for循环来实现
# 紧接着，取列表的两个集合进行比较
# 如果这两个几个的前面k-2个元素都相等，那么久将这两个集合合并成一个大小为k的集合
def aprioriGen(Lk,k):# creates Ck
    retList=[]
    lenLk=len(Lk)
    for i in range(lenLk):
        for j in range(i+1,lenLk):
            # 前k-2个项相同时，将两个集合合并
            L1=list(Lk[i])[:k-2]
            L2=list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()
            if L1==L2:
                retList.append(Lk[i]|Lk[j])
    return retList

def apriori(dataSet,minSupport=0.5):
    C1=createC1(dataSet)
    D=list(map(set,dataSet))
    L1,supportData=scanD(D,C1,minSupport)
    L=[L1]
    k=2
    while (len(L[k-2])>0):
        Ck=aprioriGen(L[k-2],k)
        Lk,supK=scanD(D,Ck,minSupport)# 扫描数据集，从Ck得到Lk
        supportData.update(supK)
        L.append(Lk)
        k+=1
    return L,supportData

L,supportData=apriori(dataSet)

# 关联规则生成函数

# 有三个参数：频繁项集列表，包含那些频繁项集支持数据的字典，最小可信度阈值
# 函数最后要生成一个包含可信度的规则列表，后面可以基于可信度对它们进行排序
# 这些规则放在bigRuleList中
# 该函数遍历L中的每一个频繁项集并对买个频繁项集创建只包含单个元素的集合列表H1
# 因为无法从单元素项集中构建关联规则，所以要从包含两个或者更多元素的项集开始规则构建过程
# 如果频繁项集的元素数目超过2，那么会考虑对其进行进一步的合并，使用函数rulesFromConseq来完成
# 如果项集中只有两个元素，那么使用函数calcConf来计算可信度
def generateRules(L,supportData,minConf=0.7):
    bigRuleList=[]
    # 只获取有两个或者更多元素的集合
    for i in range(1,len(L)):
        for freqSet in L[i]:
            H1=[frozenset([item]) for item in freqSet]
            if(i>1):
                rulesFromConseq(freqSet,H1,supportData,bigRuleList,minConf)
            else:
                calcConf(freqSet,H1,supportData,bigRuleList,minConf)
    return bigRuleList

# 计算规则的可信度，以及找到满足最小可信度要求的规则
# 创建一个空列表prunedH来保存规则
# 遍历H中的所有项集，并且计算他们的可信度
# 可信度计算时使用supportData中的书支持度数据
# 如果某条规则满足最小可信度，那么将这条规则输出到屏幕显示，
# 通过检查的规则也会被返回，并用在下一个函数rulesFromConseq中
# 同时也需要对列表br1进行填充，而br1是前面通过检查的bigRuleList
def calcConf(freqSet,H,supportData,br1,minConf=0.7):
    prunedH=[]
    for conseq in H:
        conf=supportData[freqSet]/supportData[freqSet-conseq]
        if conf>=minConf:
            print(freqSet-conseq,'-->',conseq,'conf: ',conf)
            br1.append((freqSet-conseq,conseq,conf))
            prunedH.append(conseq)
    return prunedH

# 从最初项集中生成更多的关联规则
# 两个参数，一个是频繁项集，可以出现在规则右部的元素列表H
# 函数先计算H中的频繁集大小m
# 接下来查看该频繁项集是否大到可以移除大小为m的自己
# 如果可以，将其移除
# 可以使用函数aprioriGen来生成H中元素的无重复组合
# 该结果存储在Hmp1中，也是下一次迭代的H列表
# Hmp1包含所有可能的规则
# 可以利用calcConf来测试他们的可信度以及确定规则是否满足要求
# 如果不止一条规则满足要求
# 那么使用Hmp1迭代调用函数rulesfromConseq来判断是否可以进一步组合这些规则
def rulesFromConseq(freqSet,H,supportData,br1,minConf=0.7):
    m=len(H[0])
    if(len(freqSet)>(m+1)):
        Hmp1=aprioriGen(H,m+1)
        Hmp1=calcConf(freqSet,Hmp1,supportData,br1,minConf)
        if(len(Hmp1)>1):
            rulesFromConseq(freqSet,Hmp1,supportData,br1,minConf)

rules=generateRules(L,supportData,minConf=0.5)

mushDatSet=[line.split() for line in open('mushroom.dat').readlines()]

L,supportData=apriori(mushDatSet,minSupport=0.3)

print("L:",L)
print("suppoerData:",supportData)

