#coding=utf-8
# 此脚本实现K-NN算法。大部分代码参考自《机器学习实战》《Machine Learning in Action》,作部分改动。

"""
---- author = "liang wu" ----
---- time = "20150305" ----
---- Email = "wl062345@gmail.com" ----
"""
import numpy as np
import matplotlib.pyplot as plt
import operator
import random


def readData():
    file = open("datingTestSet2.txt")
    lines = file.readlines()
    mat = np.zeros((len(lines), 3))
    index = 0
    classLabel = []
    for line in lines:
        line = line.split()
        mat[index, :] = line[0:3]
        classLabel.append(int(line[-1]))
        index += 1
    return mat, classLabel


def plotData(mat, label):
    #可视化数据
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(mat[:,0], mat[:,1], 15.0*np.array(label), 15.0*np.array(label),)
    plt.show()


def normalize(mat):
    mean = mat.mean(axis=0)   # 每一列数据均值
    dif = mat - mean
    std = mat.std(axis=0)     # 每一列数据方差
    norMat = dif/std          # 计算归一化值
    return norMat


def kNN(testRecord, trainMat , labels, k):
    numTrainMat = trainMat.shape[0]                                  # 训练数据集的行数
    dif = np.tile(testRecord, (numTrainMat, 1)) - trainMat           # tile是个特殊的模块，将testRecord复制numTrainMat行
    sq = dif**2
    distances = (sq.sum(axis=1))**0.5                                # 计算欧式距离
    sortedDistIndex = distances.argsort()                            # 是将distances按照从小到大(默认)排序的索引
    classCount = {}
    for i in range(k):
        votelabel = labels[sortedDistIndex[i]]                       # 获得按距离排序，次序在前三的标签（即离他最近的三个邻居）
        classCount[votelabel] = classCount.get(votelabel, 0) +1      # 使用get()提供默认值
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    # key指定排序元素的哪一项进行排序
    return sortedClassCount[0][0]


def main():
    [mat, label] = readData()    # 读取数据，存为矩阵
    #plotData(mat, label)        # 可视化数据
    mat = normalize(mat)         # 归一化数据
    numData = mat.shape[0]
    testRatio = 0.10
    testIndex = [int(numData*random.random()) for i in xrange(int(numData*testRatio))]
    # 一共有numData条样本，选择 numData*testRatio 条样本作为测试样本；随机选择
    testIndex = sorted(list(set(testIndex)))    # 去掉随机数中重复项

    testMat = np.zeros((len(testIndex), 3))
    trainMat = np.zeros(((numData-len(testIndex)), 3))

    testMat[:] = [mat[i,:] for i in testIndex]
    testLabel = [label[i] for i in testIndex]
    trainMat[:] = [mat[i,:] for i in xrange(numData)  if i not in testIndex]
    trainLabel = [label[i] for i in xrange(numData)  if i not in testIndex]
    errorCount = 0
    for i in range(testMat.shape[0]):
        classifierResult = kNN(testMat[i, :], trainMat[:], trainLabel, 3)
        if (classifierResult != testLabel[i]):errorCount += 1.0
    print "错误率： %f"%(errorCount/len(testIndex))


if __name__=="__main__":main()
