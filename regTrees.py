import numpy as np


def loadDataSet(fileName):
    dataList = []
    fr = open(fileName)
    for line in fr.readlines():
        currLine = line.strip().split('\t')
        fltLine = list(map(float, currLine))
        dataList.append(fltLine)
    return dataList


def linearSolve(dataSet):
    m, n = np.shape(dataSet)
    X = np.mat(np.ones((m, n)))
    Y = np.mat(np.zeros((m, 1)))
    X[:, 1:n] = dataSet[:, 0:n-1]
    Y = dataSet[:, -1]
    xTx = X.T * X
    if np.linalg.det(xTx) == 0.0:
        print("Error!")
    ws = xTx.I * (X.T * Y)
    return ws, X, Y

def modelLeaf(dataSet):
    ws, X, Y = linearSolve(dataSet)
    return ws


def modelErr(dataSet):
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return np.sum(np.power(yHat - Y, 2))


def regLeaf(dataSet):
    """
    生成叶节点
    :param dataSet:
    :return:
    """
    return np.mean(dataSet[:, -1])


def regErr(dataSet):
    """
    计算总方差
    :param dataSet:
    :return:
    """
    return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]


def binSplitDataSet(dataSet, feature, value):
    """
    切分数据集
    :param dataSet: 待切分数据集
    :param feature: 特征
    :param value: 特征取值
    :return:
    """
    dataSet = np.mat(dataSet)
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1


def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    tolS = ops[0]
    tolN = ops[1]
    dataSet = np.mat(dataSet)
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    m, n = np.shape(dataSet)
    S = errType(dataSet)
    bestS = float('inf')
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n-1):
        for splitVal in set(dataSet[:, featIndex].T.tolist()[0]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    if bestS == float('inf'):
        return None, leafType(dataSet)
    return bestIndex, bestValue



def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    """
    构建回归树
    :param dataSet: 数据集
    :param leafType: 建立叶节点的函数
    :param errType: 误差计算函数
    :param ops:
    :return:
    """
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None:
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    return retTree

def isTree(obj):
    return (type(obj).__name__ == 'dict')


def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['right'] + tree['left']) / 2

def prune(tree, testData):
    if np.shape(testData)[0] == 0:
        return getMean(tree)
    if (isTree(tree['right']) or isTree(tree['left'])):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['right']):
        prune(tree['right'], rSet)
    if isTree(tree['left']):
        prune(tree['left'], lSet)
    if not isTree(tree['right']) and not isTree(tree['left']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errNoMerge = np.sum(np.power(lSet[:, -1] - tree['left'], 2)) + np.sum(np.power(rSet[:, -1] - tree['left'], 2))
        treeMean = (tree['left'] + tree['right']) / 2
        errMerge = np.sum(np.power(testData[:, -1] - treeMean, 2))
        if errMerge < errNoMerge:
            print("Merging")
            return treeMean
        else:
            return tree
    else:
        return tree


def regTreeEval(model, inDat):
    return float(model)


def modelTreeEval(model, inDat):
    n = np.shape(inDat)[1]
    X = np.mat(np.ones((1, n+1)))
    X[:, 1:n+1] = inDat
    return float(X * model)

def treeForeCast(tree, inData, modeEval=regTreeEval):
    if not isTree(tree):
        return modeEval(tree, inData)
    if (inData[tree['spInd']] > tree['spVal']):
        return treeForeCast(tree['left'], inData, modeEval)
    else:
        return treeForeCast(tree['right'], inData, modeEval)

def createForeCast(tree, testData, modEval=regTreeEval):
    m = np.shape(testData)[0]
    yHat = np.zeros((m, 1))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, testData[i, :], modEval)
    return yHat




"""
testMat = np.mat(np.eye(4))
mat0, mat1 = binSplitDataSet(testMat, 2, 0.5)
print(mat0)
print(mat1)

myData = loadDataSet("regTrees/ex2.txt")
myDataTest = loadDataSet("regTrees/ex2test.txt")
myMat = np.mat(myData)
myMatTest = np.mat(myDataTest)
myTree = createTree(myData, ops=(0,1))
print(myTree)
myTree1 = prune(myTree, myMatTest)
print(myTree1)
"""

"""
myMat2 = np.mat(loadDataSet("regTrees/exp2.txt"))
myTree = createTree(myMat2, modelLeaf, modelErr, (1, 10))
print(myTree['left'])
"""

trainMat = np.mat(loadDataSet("regTrees/bikeSpeedVsIq_train.txt"))
testMat = np.mat(loadDataSet("regTrees/bikeSpeedVsIq_test.txt"))
myTree = createTree(trainMat, ops=(1, 20))
yHat = createForeCast(myTree, testMat[:, 0])
print(np.corrcoef(yHat, testMat[:, 1], rowvar=0))

myModelTree = createTree(trainMat, modelLeaf, modelErr, (1, 20))
yHat = createForeCast(myModelTree, testMat[:, 0], modelTreeEval)
print(np.corrcoef(yHat, testMat[:, 1], rowvar=0))

ws, X, y = linearSolve(trainMat)
yHat = []
for i in range(np.shape(testMat)[0]):
    yHat.append(testMat[i, 0] * ws[1, 0] + ws[0, 0])

print(np.corrcoef(yHat, testMat[:, 1], rowvar=0))
