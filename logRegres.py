import numpy as np
import random
import matplotlib.pyplot as plt

def loadDataSet():
    """
    载入数据集
    :return: 数据矩阵，标签矩阵
    """
    dataList = []
    labelList = []
    fr = open("logRegres/testSet.txt")
    for line in fr.readlines():
        lineList = line.strip().split()
        dataList.append([1.0, float(lineList[0]), float(lineList[1])])
        labelList.append(int(lineList[2]))
    return dataList, labelList


def sigmoid(inX):
    return 1.0 / (1.0 + np.exp(-inX))


def gradAscent(dataListIn, classLabelsList):
    """
    梯度上升
    :param dataListIn: 输入数据列表
    :param classLabels: 输入数据标签列表
    :return: 最佳回归系数
    """
    """转换为numpy格式矩阵数据类型"""

    dataMatrix = np.mat(dataListIn)
    labelMatrix = np.mat(classLabelsList).transpose()  # 矩阵转置

    m, n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n, 1)) # ndarray类型

    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = labelMatrix - h
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights


def stocGradAscent(dataListIn, classLabelsList):
    """
    随机梯度上升
    :param dataMatrix: 输入数据列表
    :param classLabels: 输入数据标签列表
    :return: 最佳回归系数
    """
    dataArr = np.array(dataListIn)
    m, n = np.shape(dataArr)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(np.sum(dataArr[i] * weights))
        error = classLabelsList[i] - h
        weights = weights + alpha * error * dataArr[i]
    return weights


def stocGradAscent1(dataListIn, classLabelsList, numIter = 150):
    """
    改进的随机梯度上升算法
    :param dataListIn: 输入数据列表
    :param classLabelsList: 输入数据标签列表
    :param numIter: 迭代次数
    :return: 最佳回归系数
    """
    dataArr = np.array(dataListIn)
    m, n = np.shape(dataArr)
    weights = np.ones(n)
    xcord = []
    ycord0 = []
    ycord1 = []
    ycord2 = []
    for j in range(numIter):
        dataArr = np.array(dataListIn)
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + i + j) + 0.1
            randomIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(np.sum(weights * dataArr[randomIndex]))
            error = classLabelsList[randomIndex] - h
            weights = weights + alpha * error * dataArr[randomIndex]
            #xcord.append(j * m + i)
            #ycord0.append(weights[0])
            #ycord1.append(weights[1])
            #ycord2.append(weights[2])
            np.delete(dataArr, randomIndex)
    """
        fig0 = plt.figure()
        ax0 = fig0.add_subplot(111)
        ax0.plot(xcord, ycord0)
        plt.xlabel("epochs")
        plt.ylabel("X0")
        plt.show()
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.plot(xcord, ycord1)
        plt.xlabel("epochs")
        plt.ylabel("X1")
        plt.show()
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        ax2.plot(xcord, ycord2)
        plt.xlabel("epochs")
        plt.ylabel("X2")
        plt.show()
    """
    return weights


def plotBestFit(weights):
    """
    画出数据集和最佳拟合直线
    :param weights: 最佳回归参数
    :return:
    """
    dataMat, labelMat = loadDataSet()
    weights = np.array(weights)
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord0 = []
    ycord0 = []
    xcord1 = []
    ycord1 = []
    for i in range(n):
        if (labelMat[i] == 0):
            xcord0.append(dataArr[i][1])
            ycord0.append(dataArr[i][2])
        else:
            xcord1.append(dataArr[i][1])
            ycord1.append(dataArr[i][2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord0, ycord0, s=30, c="red", marker="s")
    ax.scatter(xcord1, ycord1, s=30, c="green")
    X = np.arange(-3.0, 3.0, 0.1)
    Y = (-weights[0] - weights[1] * X) / weights[2]
    ax.plot(X, Y)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()

def classifyVector(inX, weights):
     """
     logistic分类器
     :param inX: 输入向量
     :param weights: 权重
     :return: 所属类别
     """
     prob = sigmoid(np.sum(inX * weights))
     if prob > 0.5:
         return 1.0
     else:
         return 0.0

def colicTest():
    """
    从疝气病症预测病马的死亡率
    :return: 错误率
    """
    frTrain = open("logRegres/horseColicTraining.txt")
    frTest = open("logRegres/horseColicTest.txt")
    trainDataList = []
    trainClassLabelsList = []
    for line in frTrain.readlines():
        currentLine = line.strip().split()
        lineList = []
        for i in range(21):
            lineList.append(float(currentLine[i]))
        trainDataList.append(lineList)
        trainClassLabelsList.append(float(currentLine[21]))
    trainWeights = stocGradAscent1(trainDataList, trainClassLabelsList, 500)
    errorCount = 0.0
    numTest = 0.0
    for line in frTest.readlines():
        numTest += 1
        currentLine = line.strip().split()
        lineList = []
        for i in range(21):
            lineList.append(float(currentLine[i]))
        if int(classifyVector(lineList, trainWeights)) != int(float(currentLine[21])):
            errorCount += 1
    errorRate = float(errorCount) / float(numTest)
    print("The error rate of this test is: %f" %errorRate)
    return errorRate

def multiTest():
    numTest = 10
    errorSum = 0.0
    for k in range(numTest):
        errorSum += colicTest()
    print("After %d iterations the average error rate is: %f" % (numTest, errorSum/float(numTest)))

"""
    dataList, labelList = loadDataSet()
    weights1 = gradAscent(dataList, labelList)
    plotBestFit(weights1)
    weights2 = stocGradAscent(dataList, labelList)
    plotBestFit(weights2)
    weights3 = stocGradAscent1(dataList, labelList, 500)
    plotBestFit(weights3)
"""
multiTest()