#coding=utf-8
# 构造树的ID3算法
from math import log
import operator

# 测试数据
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    # 这尼玛哪里是label, 明明就是特征名
    labels = ['no surfacing','flippers']
    return dataSet, labels

# 计算给定数据集的香农熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet: #the the number of unique elements and their occurance
        currentLabel = featVec[-1]
        # 标签类构建字典，记录数量
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        # 计算每种标签的概率
        prob = float(labelCounts[key])/numEntries
        # 利用概率求算香农熵
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt
    
def splitDataSet(dataSet, axis, value):
    # 待划分数据集，划分数据集的特征的index，筛选的特征的值
    # 为了不修改原数据
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]     # chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet
    
def chooseBestFeatureToSplit(dataSet):
    # 特征数，最后一列为标签
    numFeatures = len(dataSet[0]) - 1
    # 计算数据集香农熵
    baseEntropy = calcShannonEnt(dataSet)
    # 初始化
    bestInfoGain = 0.0
    bestFeature = -1
    # 迭代所有特征
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]#create a list of all the examples of this feature
        # 得到唯一的分类标签列表 牛叉的set类型 最快方法
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        # 熵的减少
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):       #compare this to the best gain so far
            bestInfoGain = infoGain         #if better than current best, set to best
            bestFeature = i
    return bestFeature                      #returns an integer

def majorityCnt(classList):
    # 字典的排序显得非常简单
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# 构建决策树
def createTree(dataSet, labels):
    # dataSet是当前例子集合
    # 类别list
    classList = [example[-1] for example in dataSet]
    # 第一个停止条件：所有类标签完全相同
    if classList.count(classList[0]) == len(classList): 
        return classList[0]
    # 第二个停止条件，使用完了所有特征
    if len(dataSet[0]) == 1:
        # 返回出现次数最多的类别作为返回值
        return majorityCnt(classList)
    # 选择最好的特征划分数据集
    bestFeat = chooseBestFeatureToSplit(dataSet)
    # 目前状态最好的特征名
    bestFeatLabel = labels[bestFeat]
    # 构建树
    myTree = {bestFeatLabel:{}}
    # 删除被选择的特征名
    del(labels[bestFeat])
    # 取出该特征的样本值
    featValues = [example[bestFeat] for example in dataSet]
    # 唯一化
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        # 贪心算法 递归
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree                            
    
def classify(inputTree, featLabels, testVec):
    # 递归函数实现字典
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel

def storeTree(inputTree, filename):
    # python里存储东西用pickle啊pickle
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()
    
def grabTree(filename):
    # 解析树
    import pickle
    fr = open(filename)
    return pickle.load(fr)

# dataSet, labels = createDataSet()
# createTree(dataSet, labels)