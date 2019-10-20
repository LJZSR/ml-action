import numpy as np
import matplotlib.pyplot as plt
import math

def loadSimpData():
    dataMat = np.mat([[1.0, 2.1],
                      [2.0, 1.1],
                      [1.3, 1.0],
                      [1.0, 1.0],
                      [2.0, 1.0]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return  dataMat, classLabels

def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = np.ones((np.shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray

def buildStump(dataArr, classLabels, D):
    """
    单层最优决策数生成
    :param dataArr: 数据
    :param classLabels: 数据类别
    :param D: 权重向量
    :return: 单层决策树，最小误差，分类结果
    """
    dataMatrix = np.mat(dataArr)
    labelMatrix = np.mat(classLabels).T
    m, n = np.shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClassEst = np.mat(np.zeros((m,1)))
    minError = float("inf")
    for i in range(n):
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, int(numSteps) + 1):
            threshVal = rangeMin + float(j) * stepSize
            for inequal in ['lt', 'gt']:
                predictedVal = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = np.mat(np.ones((m, 1)))
                errArr[predictedVal == labelMatrix] = 0
                weightedErr = D.T * errArr
                print("split: dim %d, thresh %.2f, tresh inequal: %s, the weighted error is %.3f"
                      %(i, threshVal, inequal, weightedErr))
                if weightedErr < minError:
                    minError = weightedErr
                    bestClassEst = predictedVal[:, :]
                    bestStump["dim"] = i
                    bestStump["thresh"] = threshVal
                    bestStump["ineq"] = inequal
    return bestStump, minError, bestClassEst

def adaBoostTrainDS(dataArr, classLabels, numIt = 40):
    """
    基于单层决策树的AdaBoost训练
    :param dataArr: 训练数据
    :param classLabels: 数据类别
    :param numIt: 弱分类器最大个数
    :return:
    """
    weakClassArr = []
    dataMatrix = np.mat(dataArr)
    labelMatrix = np.mat(classLabels)
    m = np.shape(dataMatrix)[0]
    D = np.mat(np.zeros((m, 1))/m)
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print("D: " + D.T)
        alpha = float(0.5 * math.log((1.0 - error) / max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print("classEst" + classEst.T)
        expon = np.multiply(-1 * )




dataMat, classLabels = loadSimpData()

D = np.mat(np.ones((5,1))/5)
print(buildStump(dataMat, classLabels, D))
"""
m, n = np.shape(dataMat)
xcord0 = []
ycord0 = []
xcord1 = []
ycord1 = []
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
for i in range(m):
    if classLabels[i] == 1:
        xcord0.append(dataMat.A[i][0])
        ycord0.append(dataMat.A[i][1])
    else:
        xcord1.append(dataMat.A[i][0])
        ycord1.append(dataMat.A[i][1])
ax.scatter(xcord0, ycord0, s=30, c="red", marker="s")
ax.scatter(xcord1, ycord1, s=30, c="green")
plt.show()

print(dataMat)
print(classLabels)
"""