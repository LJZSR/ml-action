import numpy as np
import operator
from os import listdir


def createDataSet():
    """
    创建数据集
    :return: 数据集，标签
    """
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    """
    k近邻分类器
    :param inX: 用于分类的输入向量
    :param dataSet: 输入的训练样本集合
    :param labels: 训练样本标签向量
    :param k: 最近邻居数目
    :return: 输入向量所属的类别
    """
    dataSetSize = dataSet.shape[0]  # 获取矩阵行数，即训练样本个数
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet  # 计算差值矩阵
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)  # 计算距离的平方，axis=1，按行求和
    distances = sqDistances ** 0.5  # 计算距离
    sortedDisIndicies = distances.argsort()  # 返回距离数组从小到大索引值
    classCount = {}

    # 计算k个最近点中各个标签出现的次数
    for i in range(k):
        voteIlable = labels[sortedDisIndicies[i]]
        classCount[voteIlable] = classCount.get(voteIlable, 0) + 1;

    # 对字典按值排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def file2matrix(filename):
    """
    将文本记录转换为NumPy的解析程序
    :param filename: 原始文件名
    :return: 输入矩阵、标签向量
    """
    fr = open(filename)
    arrayOLines = fr.readlines()  # 按行读取文件
    numberOfLines = len(arrayOLines)  # 获取文件行数
    returnMatrix = np.zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()  # 去除行首、行尾空字符
        listFromLine = line.split('\t')
        returnMatrix[index, :] = [float(i) for i in listFromLine[0:3]]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMatrix, classLabelVector


def img2vector(filename):
    """
    将图片格式转换为分类器使用的向量格式
    :param filename: 原始文件名
    :return: 图片的向量格式
    """
    returnVector = np.zeros((1, 1024))  # 图片格式为32*32
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVector[0, 32 * i + j] = int(lineStr[j])
    return returnVector


def autoNorm(dataSet):
    """
    归一化特征值
    :param dataSet: 没有归一化的原始数据
    :return: 归一化后的数据，原始数据极差，原始数据最小值
    """
    minVals = dataSet.min(0)  # 原始数据每一列的最小值
    maxVals = dataSet.max(0)  # 原始数据每一列的最大值
    ranges = maxVals - minVals
    dataSetSize = dataSet.shape[0]
    dataSet = dataSet - np.tile(minVals, (dataSetSize, 1))
    dataSet = dataSet / np.tile(ranges, (dataSetSize, 1))
    return dataSet, ranges, minVals


def datingClassTest():
    """
    分类器针对约会网站的测试
    """
    hoRatio = 0.1  # 前10%为测试数据
    datingDataMat, datingLabels = file2matrix("kNN/datingTestSet2.txt")
    normMat, ranges, minVals = autoNorm(datingDataMat)
    dataSetSize = normMat.shape[0]
    numTestVecs = int(dataSetSize * hoRatio)
    errorCount = 0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:dataSetSize, :],
                                     datingLabels[numTestVecs:dataSetSize], 3)
        print("The classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1
    print("The total error rate is: %f" % (errorCount / float(numTestVecs)))


def handwritingClassTest():
    """
    手写数字识别系统的测试
    """
    hwLabels = []
    trainingFileList = listdir("kNN/trainingDigits")  # 获取训练集目录文件名
    dataSizeTrain = len(trainingFileList)
    trainingMat = np.zeros((dataSizeTrain, 1024))
    for i in range(dataSizeTrain):
        filenameStr = trainingFileList[i]
        fileStr = filenameStr.split('.')[0]
        classNumStr = fileStr.split('_')[0]
        hwLabels.append(int(classNumStr))
        trainingMat[i, :] = img2vector("kNN/trainingDigits/%s" % filenameStr)
    testFileList = listdir("kNN/testDigits")  # 获取测试集目录文件名
    errorCount = 0.0
    dataSizeTest = len(testFileList)
    for i in range(dataSizeTest):
        filenameStr = testFileList[i]
        fileStr = filenameStr.split('.')[0]
        classNumStr = fileStr.split('_')[0]
        vectorUnderTest = img2vector("kNN/testDigits/%s" % filenameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("The classifier came back with: %d, the real answer is: %d" % (classifierResult, int(classNumStr)))
        if (classifierResult != int(classNumStr)):
            errorCount += 1
    print("The total error rate is: %f" % (errorCount / float(dataSizeTest)))


def classifyPerson():
    resultList = ["not at all", "in small doses", "in large doses"]
    ffMiles = float(input("Frequent flier miles earned per year?"))
    percentTats = float(input("Percentage of time spent playing video games?"))
    iceCream = float(input("Liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix("kNN/datingTestSet2.txt")
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, percentTats, iceCream])
    inArr = (inArr - minVals) / ranges
    classifierResult = classify0(inArr, normMat, datingLabels, 3)
    print("You will probably like this person: " + resultList[classifierResult - 1])
