#coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy

def loadData():
    dataMat = []
    labelMat = []
    fp = open('testSet.txt')
    for line in fp.readlines():
        lineArr = line.strip().split()
        dataMat.append([float(lineArr[0]), float(lineArr[1]), 1.0])
        labelMat.append(int(lineArr[2]))
    return np.mat(dataMat), np.mat(labelMat)

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

def plotFit():
    plt.plot()

def iniWeights(numFeature):
    weiMat = []
    for i in range(0, numFeature):
        weiMat.append(random.random())
    return np.mat(weiMat)

def trainLR(X, y, alpha, maxIter, gradMean):
    m, n = X.shape
    y = y.transpose()
    #weights = iniWeights(n).transpose()
    weights = np.ones((n, 1))
    count = 0

    if gradMean == 'BGD':
        for k in range(maxIter):
            count += 1
            h = X * weights
            predict = sigmoid(h)
            error = y - predict
            sumError = np.abs(error).sum()
            print "第",count, "次迭代.......sumError: ", sumError
            weights = weights + alpha * X.transpose() *  error

    if gradMean == 'SGD':
        for k in range(maxIter):
            count += 1
            for i in range(n):
                h = X * weights
                predict = sigmoid(h)
                error = y - predict
                temp = error
                sumError = np.abs(temp).sum()
                print "第",count, "次迭代.......sumError: ", sumError
                weights = weights + alpha * X.transpose() *  error
    return weights

def testLR(weights, X, y):
    m, n = np.shape(X)
    matchCount = 0
    predict = X*weights
    for i in xrange(m):
        predict = sigmoid(X[i, :] * weights)[0, 0] > 0.5
        if predict == bool(y[i, 0]):
            matchCount += 1
    accuracy = float(matchCount) / m
    return accuracy

def showLR(weights, X, y):
    m, n = np.shape(X)
    if n != 3:
        return 1

    for i in xrange(m):
        if int(y[i, 0]) == 0:
            plt.plot(X[i, 0], X[i, 1], 'or')
        elif int(y[i, 0]) == 1:
            plt.plot(X[i, 0], X[i, 1], 'ob')

    min_x = min(X[:, 0])[0, 0]
    max_x = max(X[:, 0])[0, 0]
    weights = weights.getA()
    y_min_x = float(- weights[2] - weights[0] * min_x) / weights[1]
    y_max_x = float(- weights[2] - weights[0] * max_x) / weights[1]
    plt.plot([min_x, max_x], [y_min_x, y_max_x], '-g')
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()


if __name__=="__main__":
    #########   加载数据   #########
    X, y = loadData()
    #########   设置参数   #########
    alpha = 0.01
    maxIter = 100
    gradMean = ['BGD', 'SGD']
    #########   训练模型   #########
    W = trainLR(X, y, alpha, maxIter, gradMean[0])
    #########   测试模型   #########
    accuracy = testLR(W, X, y.transpose())
    #########   拟合曲线   #########
    showLR(W, X, y.transpose())
