import numpy as np
import random


def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec


def textParse(bigString):
    """
    文本解析
    :param bigString: 文本字符串
    :return: 文本所含单词
    """
    import re
    listOfTokens = re.split(r'\W', bigString)  # 使用正则表达式切分文本
    return [token.lower() for token in listOfTokens if len(token) > 2]  # 仅取长度大于2的单词，并转化为小写


def createVocabList(dataSet):
    """
    建立出现单词的单词表
    :param dataSet: 数据集
    :return: 单词表
    """
    vocabSet = set()
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    """
    将文档转换为文档向量
    :param vocabList: 单词表
    :param inputSet: 文档
    :return: 文档向量（出现的单词为1，未出现的单词未0）
    """
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            i = vocabList.index(word)
            returnVec[i] = 1
        else:
            print("The word %s is not in my VocabularyList!" % word)
    return returnVec


def bagOfWords2Vec(vocabList, inputSet):
    """
    将文档转换为词袋模型
    :param vocabList: 单词表
    :param inputSet: 文档
    :return: 文档的词袋模型
    """
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            i = vocabList.index(word)
            returnVec[i] += 1
        else:
            print("The word %s is not in my VocabularyList!" % word)
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    """
    训练朴素贝叶斯分类器
    :param trainMatrix: 样本矩阵
    :param trainCategory: 每个样本所属类别
    :return: p(w/ci)向量, p(ci)
    """
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = np.ones(numWords)  # 防止出现0
    p0Denom = 2.0
    p1Num = np.ones(numWords)  # 防止出现0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p0Vector = np.log(p0Num / p0Denom)
    p1Vector = np.log(p1Num / p1Denom)
    return p0Vector, p1Vector, pAbusive


def classifyNB(vec2Classify, p0V, p1V, pClass1):
    """
    朴素贝叶斯分类器
    :param vec2Classify: 待分类向量
    :param p0V: p(w|c0)向量
    :param p1V: p(w|c1)向量
    :param pClass1: p(c1)
    :return: 待分类向量所属类别
    """
    p1 = sum(vec2Classify * p1V) + pClass1
    p0 = sum(vec2Classify * p0V) + (1 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    """
    测试
    :return: 测试结果
    """
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMatrix = []
    for document in listOPosts:
        trainMatrix.append(setOfWords2Vec(myVocabList, document))
    p0V, p1V, pAb = trainNB0(trainMatrix, listClasses)
    testEntry = ["love", "my", "dalmation"]
    thisDoc = setOfWords2Vec(myVocabList, testEntry)
    print(str(testEntry) + " classified as: %d" % classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ["stupid", "garbage"]
    thisDoc = setOfWords2Vec(myVocabList, testEntry)
    print(str(testEntry) + " classified as: %d" % classifyNB(thisDoc, p0V, p1V, pAb))


def spamTest():
    docList = []
    classList = []
    fullText = []
    """导入并解析文本文件"""
    for i in range(1, 26):
        wordList = textParse(open("bayes/email/spam/%d.txt" % i, encoding="ISO-8859-1").read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open("bayes/email/ham/%d.txt" % i, encoding="ISO-8859-1").read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)

    """随机构建训练集合"""
    trainingSet = list(range(50))
    testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])

    """训练"""
    trainMat = []
    trainClass = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClass.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(trainMat, trainClass)

    """测试"""
    errorCount = 0.0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(wordVector, p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print(docList[docIndex])
    print("The error rate is: %f" % (errorCount / float(len(testSet))))


spamTest()
