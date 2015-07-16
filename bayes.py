#coding=utf-8
# 假设特征之间相互独立，这个假设正是朴素贝叶斯分类器中朴素(naive)一次的含义，另一个假设是特征同等重要
from numpy import *
import feedparser
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 代表侮辱性文字, 0 代表正常言论
    return postingList, classVec
                 
def createVocabList(dataSet):
    #  一个无序不重复元素集, 基本功能包括关系测试和消除重复元素. 集合对象还支持union(联合),
    #  intersection(交), difference(差)和sysmmetric difference(对称差集)等数学运算
    #  sets 支持 x in set, len(set),和 for x in set。
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document) #union of the two sets
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print "the word: %s is not in my Vocabulary!" % word
    return returnVec

# 以上三个函数所做的事情是：已经知道一个词是否出现在一篇文档中，也知道该文档所属的类别

def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)    # 样本数
    numWords = len(trainMatrix[0])     # 特征数
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # 侮辱性标签所占的概率
    p0Num = ones(numWords)   # 初始化个数向量，为了避免乘积为零
    p1Num = ones(numWords)
    p0Denom = 2.0            # 初始化总的个数
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 最后 p1num 是指在标签值为侮辱性的情况下的每个特征出现的个数，是个矢量 p1num 同理
    # 最后 p1Denom 是指标签为侮辱性的情况下的所有特征出现的个数
    p1Vect = log(p1Num / p1Denom)
    p0Vect = log(p0Num / p0Denom)    # 通过求解自然对数避免下溢或者浮点数舍入导致的错误
    return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0
    
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1     # 计数模型
    return returnVec

def testingNB():
    listOPosts, listClasses = loadDataSet()    # X, y
    myVocabList = createVocabList(listOPosts)  # 文本特征
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    # trainMat即为样本转化而来的0-1数值样本
    # array(trainMat), array(listClasses) 分别为 X, y
    # 得到后验概率
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    # 输入语料
    testEntry = ['love', 'my', 'dalmation']
    # 转化为特征向量
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    # 分类结果
    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)

# 每个词是否出现作为一个特征，这叫做“词集模型”
# 词袋模型则是有单词出现个数

def textParse(bigString):    #input is big string, #output is word list
    import re
    listOfTokens = re.split(r'\W*', bigString)   # 分隔符是除了单词和数字外任意字符串
    return [tok.lower() for tok in listOfTokens if len(tok) > 2] # 清除长度小于2的item，并且转化为小写
    
def spamTest():
    docList=[]; classList = []; fullText =[]
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)    # 分样本
        fullText.extend(wordList)   # 合并
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    vocabList = createVocabList(docList)     # 构建特征
    trainingSet = range(50)
    testSet=[]
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))  # 随机数的使用，随机挑选十个样本作为测试样本
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[]          # 训练矩阵 X
    trainClasses = []    # 样本类别 y
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0

    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
            print "classification error", docList[docIndex]
    print 'the error rate is: ',float(errorCount)/len(testSet)

def calcMostFreq(vocabList, fullText):
    # 计算频率最高的三十个词汇
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1), reverse=True) 
    return sortedFreq[:30]       

def localWords(feed1, feed0):
    docList=[]
    classList = []
    fullText =[]
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)   # 纽约分类为1
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    vocabList = createVocabList(docList)   # 构建词汇特征
    top30Words = calcMostFreq(vocabList, fullText)   #  频率最高的30个词汇
    print top30Words
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])
    trainingSet = range(2*minLen)
    testSet=[]
    for i in range(5):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat),array(trainClasses))

    errorCount = 0
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print 'the error rate is: ', float(errorCount)/len(testSet)
    return vocabList,p0V,p1V

def getTopWords(ny,sf):
    import operator
    vocabList,p0V,p1V=localWords(ny,sf)
    topNY=[]; topSF=[]
    for i in range(len(p0V)):
        if p0V[i] > -6.0 : topSF.append((vocabList[i],p0V[i]))
        if p1V[i] > -6.0 : topNY.append((vocabList[i],p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print "SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**"
    for item in sortedSF:
        print item[0]
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print "NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**"
    for item in sortedNY:
        print item[0]

ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
localWords(ny, sf)