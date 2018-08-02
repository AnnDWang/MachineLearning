class treeNode:
    def __init__(self,nameValue,numOccur,parentNode):
        self.name=nameValue
        self.count=numOccur
        # 用于链接相似的元素项
        self.nodeLink=None
        self.parent=parentNode
        self.children={}

    # 对count变量增加给定值
    def inc(self,numOccur):
        self.count+=numOccur

    # 用于将树以文本形式显示
    def disp(self,ind=1):
        print('   '*ind,self.name,'   ',self.count)
        for child in self.children.values():
            child.disp(ind+1)

# rootNode=treeNode('pyramid',9,None)
# rootNode.children['eye']=treeNode('eye',13,None)
# rootNode.disp()

# FP树构建函数

# 2个参数：数据集，最小支持度
# 第一遍遍历扫描数据集并统计每个元素项出现的频度
# 这些信息被存储在头指针表中
# 接下来，扫描头指针表，删掉哪些出现次数少于minSup的项
# 如果所有项都不频繁，就不需要进行下一步处理
# 接下来，对头指针表稍加扩展以便可以保存计数值及指向每种类型第一个元素项的指针
# 然后创建只包含空集合的根节点
# 最后，再遍历一次数据集，这次只考虑哪些频繁项
# 这些项已经经过排序
# 然后调用updateTree方法
def createTree(dataSet,minSup=1):
    headerTable={}
    for trans in dataSet:
        for item in trans:
            headerTable[item]=headerTable.get(item,0)+dataSet[trans]
    # 移除不满足最小支持度的元素项
    for k in list(headerTable.keys()):
        if headerTable[k]<minSup:
            del(headerTable[k])

    freqItemSet=set(headerTable.keys())
    # 如果没有元素项满足要求，则退出
    if len(freqItemSet)==0: return None,None
    for k in headerTable:
        headerTable[k]=[headerTable[k],None]
    retTree=treeNode('Null Set',1,None)
    for tranSet,count in dataSet.items():
        localD={}
        for item in tranSet:
            # 根据全局频率对每个事物中的元素进行排序
            if item in freqItemSet:
                localD[item]=headerTable[item][0]
        if len(localD)>0:
            orderedItems=[v[0] for v in sorted(localD.items(),key=lambda  p:p[1],reverse=True)]
            updateTree(orderedItems,retTree,headerTable,count)# 使用排序后的频率项集对树进行田中
    return retTree,headerTable

# 输入参数：一个项集
# 首先测试事务中的第一个元素项是否作为子节点存在
# 如果存在，更新该元素项的技术
# 如果不存在，则创建一个新的treeNode并将其作为一个子节点添加到树种
# 这时，头指针也要更新以指向新的节点
# 更新头指针表需要调用函数updateHeader
# 接下来，会讨论该函数的细节
# updateTree完成的最后一件事实不断迭代调用自身
# 每次调用会去掉列表中的第一个元素
def updateTree(items,inTree,headerTable,count):
    if items[0] in inTree.children:
        inTree.children[items[0]].inc(count)
    else:
        inTree.children[items[0]]=treeNode(items[0],count,inTree)
        if headerTable[items[0]][1]==None:
            headerTable[items[0]][1]=inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1],inTree.children[items[0]])
    if len(items)>1:
        # 对剩下的元素迭代调用updateTree函数
        updateTree(items[1::],inTree.children[items[0]],headerTable,count)

# 它确保节点链接指向树中该元素项的每一个实例
# 从头指针表的nodeLink开始，一直沿着nodeLink直到到达链表末尾
# 这就是一个链表
# 当处理树的时候，一种很自然的反应就是迭代完成每一件事
# 当以相同的方式处理链表时可能会遇到一些问题
# 链表很长可能会遇到迭代调用的次数限制
def updateHeader(nodeToTest,targetNode):
    while(nodeToTest.nodeLink!=None):
        nodeToTest=nodeToTest.nodeLink
    nodeToTest.nodeLink=targetNode

# 简单数据集及数据包装器
def loadSimpDat():
    simpDat=[['r','z','h','j','p'],
             ['z','y','x','w','v','u','t','s'],
             ['z'],
             ['r','x','n','o','s'],
             ['y','r','x','z','q','t','p'],
             ['y','z','x','e','q','s','t','m']]
    return simpDat

def createInitSet(dataSet):
    retDict={}
    for trans in dataSet:
        retDict[frozenset(trans)]=1
    return retDict

simDat=loadSimpDat()
initSet=createInitSet(simDat)

myFPtree,myHeaderTab=createTree(initSet,3)
myFPtree.disp()

# 发现以给定元素项结尾的所有路径的函数
def ascendTree(leafNode,prefixPath):
    # 迭代上溯整颗树
    if leafNode.parent!=None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent,prefixPath)

# 为给定元素项生成一个条件模式基
# 通过访问树中所有包含给定元素项的节点来完成
# 当创建树的时候，使用头指针表来指向该类型的第一个元素项
# 该元素项也会链接到其后续元素项
# 函数findPrefixPath遍历链表直到到达结尾
# 每一个元素都用ascendTree来上溯FP树，并收集所有遇到的元素项的名称
# 该列表返回之后添加到条件模式基字典condPats
def findPrefixPath(basePat,treeNode):
    condPats={}
    while treeNode!=None:
        prefixPath=[]
        ascendTree(treeNode,prefixPath)
        if len(prefixPath)>1:
            condPats[frozenset(prefixPath[1:])]=treeNode.count
        treeNode=treeNode.nodeLink
    return condPats

# 递归查找频繁项集的mineTree函数

# 首先对头指针表中的元素项按照其出现的频率进行排序，默认顺序是从小到大
# 然后，将每一个频繁项添加到频繁项集列表freqItemList中，接下来
def mineTree(inTree,headerTable,minSup,preFix,freqItemList):
    bigL=[v[0] for v in sorted(headerTable.items(),key=lambda p:p[1][0])]
    # 从头指针表的底端开始
    for basePat in bigL:
        newFreqSet=preFix.copy()
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet)
        condPattBases=findPrefixPath(basePat,headerTable[basePat][1])
        # 从条件模式基来构建条件FP树
        myCondTree,myHead=createTree(condPattBases,minSup)
        # 挖掘条件FP树
        if myHead!=None:
            print('conditional tree for: ',newFreqSet)
            myCondTree.disp(1)
            mineTree(myCondTree,myHead,minSup,newFreqSet,freqItemList)

freqItems=[]
mineTree(myFPtree,myHeaderTab,3,set([]),freqItems)