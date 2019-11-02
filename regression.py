import numpy as np
import matplotlib.pyplot as plt


def loadDataSet(filename):
    numFeat = len(open(filename).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

def standRegress(xArr, yArr):
    """
    线性回归
    :param xArr:
    :param yArr:
    :return:
    """
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws


def lwlr(testPoint, xArr, yArr, k = 1.0):
    """
    局部加权线性回归
    :param testPoint:
    :param xArr:
    :param yArr:
    :param k:
    :return:
    """
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    m = np.shape(xArr)[0]
    weights = np.mat(np.eye((m)))
    for i in range(m):
        diffMat = testPoint - xMat[i, :]
        weights[i, i] = np.exp(diffMat * diffMat.T / (-2 * k ** 2))
        xTx = xMat.T * weights * xMat
        if np.linalg.det(xTx) == 0.0:
            print("This matrix is singular, cannot do inverse")
            return
        ws = xTx.I * (xMat.T * weights * yMat)
    return testPoint * ws

def lwlrTest(testArr, xArr, yArr, k = 1.0):
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat

def rssError(yArr, yHatArr):
    return ((yArr - yHatArr) ** 2).sum()


def ridgeRegress(xArr, yArr, lam=0.2):
    """
    岭回归
    :param xArr:
    :param yArr:
    :param lam:
    :return:
    """
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    xTx = xMat.T * xMat
    denom = xTx + np.eye(np.shape(xMat)[1]) * lam
    if np.linalg.det(denom) == 0.0:
        print("This matrix i signular, cannot do inverse")
    else:
        ws = denom.I * (xMat.T * yMat)
    return ws

def ridgeTest(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean
    xMean = np.mean(xMat, 0)
    xVar = np.var(xMat, 0)
    xMat = (xMat - xMean) / xVar
    numTestPts = 30
    wMat = np.zeros((numTestPts, np.shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegress(xMat, yMat, np.exp(i-10))
        wMat[i, :] = ws.T
    return wMat

def stageWise(xArr, yArr, eps=0.1, numIt=100):
    """
    前向逐步回归
    :param xArr:
    :param yArr:
    :param eps:
    :param numIt:
    :return:
    """
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean
    xMean = np.mean(xMat, 0)
    xVar = np.var(xMat, 0)
    xMat = (xMat - xMean) / xVar
    m, n = np.shape(xMat)
    returnMat = np.zeros((numIt, n))
    ws = np.zeros((n, 1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):
        print(ws.T)
        lowestErr = float("inf")
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += sign * eps
                yTest = xMat * wsTest
                rssE = rssError(yMat.A, yTest.A)
                if rssE < lowestErr:
                    lowestErr = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T
    return returnMat



"""
xArr, yArr = loadDataSet("regression/ex0.txt")
xMat = np.mat(xArr)
yHat = lwlrTest(xArr, xArr, yArr, 0.003)
srtInd = xMat[:, 1].argsort(0)
xSort = xMat[srtInd][:, 0, :]
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(xSort[:, 1], yHat[srtInd])
ax.scatter(xMat[:, 1].flatten().A[0], np.mat(yArr).T.flatten().A[0], s=2, c='red')
plt.show()
abX, abY = loadDataSet("regression/abalone.txt")
yHat01 = lwlrTest(abX[:99], abX[:99], abY[:99], 0.1)
yHat1 = lwlrTest(abX[:99], abX[:99], abY[:99], 1)
yHat10 = lwlrTest(abX[:99], abX[:99], abY[:99], 10)
print(rssError(abY[:99], yHat01))
print(rssError(abY[:99], yHat1))
print(rssError(abY[:99], yHat10))
"""

"""
abX, abY = loadDataSet("regression/abalone.txt")
ridgeWeights = ridgeTest(abX, abY)
print(ridgeWeights)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(ridgeWeights)
plt.show()
"""

xArr, yArr = loadDataSet("regression/abalone.txt")
weights = stageWise(xArr, yArr, 0.001, 5000)
xMat = np.mat(xArr)
yMat = np.mat(yArr).T
yMean = np.mean(yMat, 0)
yMat = yMat - yMean
xMean = np.mean(xMat, 0)
xVar = np.var(xMat, 0)
xMat = (xMat - xMean) / xVar
print(standRegress(xMat, yMat.T).T)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(weights)
plt.show()
