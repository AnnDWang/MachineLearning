from numpy import *
import feedparser
# 词表到向量的转换函数
def loadDataSet():
    postingList=[['my','dog','has','flea','problems','help','please'],
                 ['maybe','not','take','him','to','dog','park','stupid'],
                 ['my','dalmation','is','so','cute','I','love','him'],
                 ['stop','posting','stupid','worthless','garbage'],
                 ['mr','licks','ate','my','steak','how','to','stop','him'],
                 ['quit','buying','worthless','dog','food','stupid']]
    classVec=[0,1,0,1,0,1]# 1代表侮辱性文字，0代表正常言论
    return postingList,classVec

# 创建一个包含在所有文档中出现的不重复词的列表，为此使用了python的set数据类型
# 将词条列表输给set构造函数，set就会返回一个不重复词表
# 首先创建一个空集合，然后将每篇文档返回的新词集合添加到该集合中。
# 操作符|用于求两个集合的并集，这也是一个按位或（OR）操作符
# 在数学符号表示上，按位或操作与集合求并操作使用相同记号
def createVocabList(dataSet):
    vocabSet=set([])# 创建一个空集
    for document in dataSet:
        vocabSet=vocabSet|set(document) # 创建两个集合的并集
    return list(vocabSet)

# 输入参数为词汇表及某个文档
# 输出时文档向量
# 向量的每个元素为1或0，分别表示词汇表的单词在输入文档中是否出现
# 函数首先创建一个和词汇表等长的向量，并将其元素都设置为0
# 接着，遍历文档中的所有单词，如果出现了词汇表中的单词，则将输出的文档向量中的对应值设为1
# 一切都顺利，就不需要检查某个词是否还在vocabList中
def setOfWords2Vec(vocabList,inputSet):
    returnVec=[0]*len(vocabList) # 创建一个其中所含元素都为0的向量
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1
        else:
            print("the word: %s is not in my Voabulary!" % word)
    return returnVec

# 朴素贝叶斯词袋模型
def bagOfWords2VecMN(vocabList,inputSet):
    returnVec=[0]* len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]+=1
    return returnVec

listOPosts,listClasses=loadDataSet()

myVocabList=createVocabList(listOPosts)

myreturnVec=setOfWords2Vec(myVocabList,listOPosts[0])

# 朴素贝叶斯分类器训练函数
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs=len(trainMatrix)
    numWords=len(trainMatrix[0])
    pAbusive=sum(trainCategory)/float(numTrainDocs)
    p0Num=ones(numWords)
    p1Num=ones(numWords)
    p0Denom=2.0
    p1Denom=2.0  # 初始化概率
    for i in range(numTrainDocs):
        if trainCategory[i]==1:
            p1Num+=trainMatrix[i]   # 向量相加
            p1Denom+=sum(trainMatrix[i])
        else:
            p0Num+=trainMatrix[i]
            p0Denom+=sum(trainMatrix[i])
    p1Vect=log(p1Num/p1Denom)  # change to log()
    p0Vect=log(p0Num/p0Denom)  # change to log()
    return p0Vect,p1Vect,pAbusive

trainMat=[]
for postinDoc in listOPosts:
    trainMat.append(setOfWords2Vec(myVocabList,postinDoc))

p0v,p1v,pAb=trainNB0(trainMat,listClasses)

# 朴素贝叶斯分类函数
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1=sum(vec2Classify*p1Vec)+log(pClass1)
    p0=sum(vec2Classify*p0Vec)+log(1.0-pClass1)

    if p1>p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts,listClasses=loadDataSet()
    myVocabList=createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0V,p1V,pAb=trainNB0(array(trainMat),array(listClasses))
    testEntry=['love','my','dalmation']
    thisDoc=array(setOfWords2Vec(myVocabList,testEntry))
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry=['stupid','garbage']
    thisDoc=array(setOfWords2Vec(myVocabList,testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))


testingNB()

# 切分文本
mySent='this book is the best book on python or M.L. I have ever laid eyes upon'
mySent.split()

# 正则切分
import re
regEx=re.compile('\\W*') # 切分分隔符是除单词、数字外的任意字符串
listOfTokens=regEx.split(mySent)

# 需要去除其中的空字符串
[tok for tok in listOfTokens if len(tok)>0]

# 转换大小写
[tok.lower() for tok in listOfTokens if len(tok)>0]

emailText=open('email/ham/6.txt',encoding='gbk').read()
listOfTokens=regEx.split(emailText)

# 文本解析及完整的垃圾邮件测试函数
def textParse(bigString):
    import re
    listOfTokens=re.split(r'\W*',bigString)
    return [tok.lower() for tok in listOfTokens if len(tok)>2]

def spamTest():
    docList=[]
    classList=[]
    fullText=[]
    for i in range(1,26):
        wordList=textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(1)
        wordList=textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList=createVocabList(docList)
    trainingSet=range(50)
    testSet=[]
    for i in range(10):
        randIndex=int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])

    trainMat=[]
    trainClasses=[]

    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList,docList[docList]))
        trainClasses.append(classList[docIndex])

    p0V,p1V,pSpam=trainNB0(array(trainMat),array(trainClasses))
    errorCount=0
    for docIndex in testSet:
        wordVector= setOfWords2Vec(vocabList,docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam)!=classList[docIndex]:
            errorCount+=1
    print('the error rate is :',float(errorCount)/len(testSet))


# RSS源分类器及高频词去除函数
def calMostFreq(vocabList,fullText):
    # 计算出现频率
    import operator
    freqDict={}
    for token in vocabList:
        freqDict[token]=fullText.count(token)
    sortedFreq=sorted(freqDict.items(),key=operator.itemgetter(1),reverse=True)
    return sortedFreq[:30]

def localWords(feed1,feed0):

    docList=[]
    classList=[]
    fullText=[]
    minLen=min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        wordList=textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList=textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList=createVocabList(docList)
    top30Words=calMostFreq(vocabList,fullText)
    # 去掉那些出现次数最高的词
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    trainingSet=range(2*minLen)
    testSet=[]
    for i in range(20):
        randIndex=int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[]
    trainClasses=[]
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam=trainNB0(array(trainMat),array(trainClasses))
    errorCount=0
    for docIndex in testSet:
        wordVector=bagOfWords2VecMN(vocabList,docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam)!=classList[docIndex]:
            errorCount+=1
    print('the error rate is: ',float(errorCount)/len(testSet))
    return vocabList,p0V,p1V



