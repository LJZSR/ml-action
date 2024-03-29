import random
import numpy as np
import matplotlib.pyplot as plt


def loadDataSet(fileName):
    """
    从文本导入数据
    :param fileName: 文本路径
    :return: 数据列表，类标签列表
    """
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


def selectJrand(i, m):
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


def clipAplha(aj, H, L):
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    """
    简化的SMO算法
    :param dataMatIn: 输入数据列表
    :param classLabels: 类别列表
    :param C: 常数C
    :param toler: 容错率
    :param MaxIter: 最大循环次数
    :return: alpha, b
    """
    dataMatrix = np.mat(dataMatIn)
    labelMatrix = np.mat(classLabels).transpose()
    b = 0
    m, n = np.shape(dataMatrix)
    alphas = np.mat(np.zeros((m, 1)))
    iter = 0
    while (iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(np.multiply(alphas, labelMatrix).T * (dataMatrix * dataMatrix[i, :].T)) + b
            Ei = fXi - float(labelMatrix[i])
            if ((labelMatrix[i] * Ei < -toler) and (alphas[i] < C)) or (
                    (labelMatrix[i] * Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i, m)
                fXj = float(np.multiply(alphas, labelMatrix).T * (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMatrix[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if labelMatrix[i] != labelMatrix[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[i] + alphas[j] - C)
                    H = min(C, alphas[i] + alphas[j])
                if L == H:
                    print("L == H")
                    continue
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - dataMatrix[i, :] * dataMatrix[i, :].T - dataMatrix[
                                                                                                            j,
                                                                                                            :] * dataMatrix[
                                                                                                                 j, :].T
                if eta >= 0.0:
                    print("eta >= 0")
                    continue
                alphas[j] -= labelMatrix[j] * (Ei - Ej) / eta
                alphas[j] = clipAplha(alphas[j], H, L)
                if abs(alphas[j] - alphaJold) < 0.00001:
                    print("j not moving enough")
                    continue
                alphas[i] += labelMatrix[j] * labelMatrix[i] * (alphaJold - alphas[j])
                b1 = b - Ei - labelMatrix[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[i, :].T - \
                     labelMatrix[j] * (alphas[j] - alphaJold) * dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMatrix[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j, :].T - \
                     labelMatrix[j] * (alphas[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[j, :].T
                if alphas[i] > 0:
                    b = b1
                elif alphas[j] > 0:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print("iter: %d i: %d pairs changed: %d" % (iter, i, alphaPairsChanged))
        if alphaPairsChanged == 0:
            iter += 1
        else:
            iter = 0
        print("iteration number: %d" % iter)
    return b, alphas


def kernelTrans(X, A, kTup):
    m, n = np.shape(X)
    K = np.zeros((m, 1))
    if kTup[0] == 'lin':
        K = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
            K[j] = np.exp(K[j] / (-1 * kTup[1]**2))
    else:
        raise NameError("That kernel is not recognized")
    return K

class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        self.X = dataMatIn
        self.labelMatrix = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2)))
        self.K = np.mat(np.zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)


def calcEk(oS, k):
    fXk = float(np.multiply(oS.alphas, oS.labelMatrix).T * oS.K[:, k] + oS.b)
    Ek = fXk - float(oS.labelMatrix[k])
    return Ek


def selectJ(i, oS, Ei):
    maxJ = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]
    validEcacheList = np.nonzero(oS.eCache[:, 0].A)[0]
    if len(validEcacheList) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ej)
            if deltaE > maxDeltaE:
                maxDeltaE = deltaE
                maxJ = k
                Ej = Ek
        return maxJ, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
        return j, Ej


def updataEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]


def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMatrix[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or (
            (oS.labelMatrix[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if oS.labelMatrix[i] != oS.labelMatrix[j]:
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[i] + oS.alphas[j] - oS.C)
            H = min(oS.C, oS.alphas[i] + oS.alphas[j])
        if L == H:
            print("L == H")
            return 0
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]
        if eta >= 0.0:
            print("eta >= 0")
            return 0
        oS.alphas[j] -= oS.labelMatrix[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAplha(oS.alphas[j], H, L)
        updataEk(oS, j)
        if abs(oS.alphas[j] - alphaJold) < 0.00001:
            print("j not moving enough")
            return 0
        oS.alphas[i] += oS.labelMatrix[j] * oS.labelMatrix[i] * (alphaJold - oS.alphas[j])
        updataEk(oS, i)
        b1 = oS.b - Ei - oS.labelMatrix[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] - \
             oS.labelMatrix[j] * (oS.alphas[j] - alphaJold) * oS.K[i, j]
        b2 = oS.b - Ej - oS.labelMatrix[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - \
             oS.labelMatrix[j] * (oS.alphas[j] - alphaJold) * oS.K[j, j]
        if oS.alphas[i] > 0:
            oS.b = b1
        elif oS.alphas[j] > 0:
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 0
    else:
        return 1

def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup = ('lin', 1)):
    """
    完整版 Platt SMO 算法
    :param dataMatIn:
    :param classLabels:
    :param C:
    :param toler:
    :param maxIte:
    :param kTup:
    :return:
    """
    oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler, kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or entireSet):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
                print("fullSet, iter: %d, i: %d, pair changed: %d" %(iter, i, alphaPairsChanged))
            iter += 1
        else:
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print("non-bound, iter: %d, i: %d, pair changed: %d" %(iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False
        elif alphaPairsChanged == 0:
            entireSet = True
        print("iteration number: %d" %iter)
    return oS.b, oS.alphas

def testRbf(k1 = 1.3):
    dataArr, labelArr = loadDataSet("svm/testSetRBF.txt")
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))
    datMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    svInd = np.nonzero(alphas.A>0)[0]
    sVs = datMat[svInd]
    labelSV = labelMat[svInd]
    print("There are %d Support Vectors" %(np.shape(labelSV)[0]))
    m, n = np.shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], ('rbf', k1))
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    print("the training error rate is: %f" % (float(errorCount) / m))
    dataArr, labelArr = loadDataSet("svm/testSetRBF2.txt")
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    errorCount = 0
    m, n = np.shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], ('rbf', k1))
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    print("the test error rate is: %f" % (float(errorCount) / m))

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

def loadImage(dirName):
    from os import listdir
    hwLabels = []
    traingFileList = listdir(dirName)
    m = len(traingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = traingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        className = int(fileStr.split('_')[0])
        if className == 9:
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        trainingMat[i, :] = img2vector("%s/%s" %(dirName, fileNameStr))
    return trainingMat, hwLabels

def testDigits(kTup=('rbf', 10)):
    dataArr, labelArr = loadImage("kNN/trainingDigits")
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    datMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    svInd = np.nonzero(alphas.A > 0)[0]
    sVs = datMat[svInd]
    labelSV = labelMat[svInd]
    print("There are %d Support Vectors" % (np.shape(labelSV)[0]))
    m, n = np.shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], kTup)
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    print("the training error rate is: %f" % (float(errorCount) / m))
    dataArr, labelArr = loadImage("kNN/testDigits")
    datMat = np.mat(dataArr)
    errorCount = 0
    m, n = np.shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], kTup)
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    print("the test error rate is: %f" % (float(errorCount) / m))



# dataList, labelList = loadDataSet("svm/testSet.txt")
# b, alphas = smoP(dataList, labelList, 0.6, 0.001, 40)
# xcord0 = []
# ycord0 = []
# xcord1 = []
# ycord1 = []
# dataNum = len(dataList)
# for i in range(dataNum):
#     if labelList[i] < 0:
#         xcord0.append(dataList[i][0])
#         ycord0.append(dataList[i][1])
#     else:
#         xcord1.append(dataList[i][0])
#         ycord1.append(dataList[i][1])
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(xcord0, ycord0, s=30, c="red", marker="s")
# ax.scatter(xcord1, ycord1, s=30, c="green")
# for i in range(dataNum):
#     if alphas[i] > 0:
#         plt.scatter(dataList[i][0], dataList[i][1], color='', marker='o', edgecolors='g', s=200)
#
# X = np.arange(2.0, 6.0, 0.1)
#
# A = 0.0
# B = 0.0
# for i in range(dataNum):
#     A += alphas[i] * labelList[i] * dataList[i][0]
#     B += alphas[i] * labelList[i] * dataList[i][1]
#
# Y = (-b - A * X) / B
# Y = np.array(Y)
# Y = Y.reshape((40,))
# ax.plot(X, Y)
# plt.xlabel("X1")
# plt.ylabel("X2")
# plt.show()

testDigits()