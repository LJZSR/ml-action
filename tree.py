from math import log
import operator
import copy

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        classCount[vote] = classCount.get(vote, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=true)
    return sortedClassCount[0][0]


def createDataSet():
    """
    创建数据集
    :return: 数据集，标签
    """
    dataSet = [[1, 1, "yes"],
               [1, 1, "yes"],
               [1, 0, "no"],
               [0, 1, "no"],
               [0, 1, "no"]]
    labels = ["no surfacing", "flippers"]
    return dataSet, labels


def calcShannonEnt(dataSet):
    """
    计算数据集的香农墒
    :param dataSet: 给定数据集
    :return: 香农墒
    """
    numEntries = len(dataSet)
    labelCount = {}
    for i in range(numEntries):
        currentLabel = dataSet[i][-1]
        labelCount[currentLabel] = labelCount.get(currentLabel, 0) + 1
    shannonEnt = 0.0
    for value in labelCount.values():
        prob = float(value) / float(numEntries)
        shannonEnt += -prob * log(prob, 2)
    return shannonEnt


def splitDataSet(dataSet, axis, value):
    """
    按照给定特征划分数据集
    :param dataSet: 待划分数据集
    :param axis: 划分第几个特征
    :param value: 特征的取值
    :return: 特征满足要求的数据集
    """
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = []
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    """
    选择最佳数据集划分方式
    :param dataSet: 带划分数据集
    :return: 最佳划分特征
    """
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = float(len(subDataSet)) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestFeature = i
            bestInfoGain = infoGain
    return bestFeature


def createTree(dataSet, labels):
    """
    递归创建决策树
    :param dataSet: 数据集
    :param labels: 分类标签
    :return: 决策树
    """
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    try:
        bestFeat = chooseBestFeatureToSplit(dataSet)
        bestFeatLabel = labels[bestFeat]
    except IndexError:
        print(dataSet)
    myTree = {bestFeatLabel: {}}
    del labels[bestFeat]
    featList = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featList)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


def classify(inputTree, featLabels, testVec):
    """
    决策树分类函数
    :param inputTree: 决策树
    :param featLabels: 特征标签
    :param testVec: 待分类向量
    :return: 所属类别
    """
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in list(secondDict.keys()):
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == "dict":
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


def storeTree(inputTree, filename):
    """
    存储决策树
    :param inputTree: 待存储树
    :param filename: 文件名
    :return:
    """
    import pickle
    fw = open(filename, "wb")
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    """
    读取树
    :param filename: 文件名
    :return: 决策树
    """
    import pickle
    fr = open(filename, "rb")
    return pickle.load(fr)


def file2dataSet(filename):
    """
    将文件转化为可以生成决策树的格式
    :param filename: 文件名
    :return: 数据集，标签向量
    """
    fr = open(filename)
    dataSet = []
    for inst in fr.readlines():
        dataSet.append(inst.strip().split("\t"))
    labels = ["age", "prescript", "astigmatic", "tearRate"]
    fr.close()
    return dataSet, labels
