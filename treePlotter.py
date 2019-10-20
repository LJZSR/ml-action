import matplotlib.pyplot as plt

"""定义文本框和箭头格式"""
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_argw = dict(arrowstyle="<-")


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    """
    绘制带箭头的注解
    :param nodeTxt: 注解文字
    :param centerPt: 注解位置
    :param parentPt: 箭头起点
    :param nodeType: 注解格式
    :return:
    """
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords="axes fraction",
                            xytext=centerPt, textcoords="axes fraction",
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_argw)


def plotMidText(cntrPt, parentPt, txtString):
    """
    在父子节点之间填充文本信息
    :param cntrPt: 子节点坐标
    :param parentPt: 父节点坐标
    :param txtString: 需填充信息
    :return:
    """
    xMid = (parentPt[0] + cntrPt[0]) / 2.0
    yMid = (parentPt[1] + cntrPt[1]) / 2.0
    createPlot.ax1.text(xMid, yMid, txtString)

def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff -= 1.0/plotTree.totalD
    for key in list(secondDict.keys()):
        if type(secondDict[key]).__name__ == "dict":
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff += 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff += 1.0 / plotTree.totalD


def createPlot(inTree):
    """
    绘图
    :return:
    """
    fig = plt.figure(1, facecolor="white")
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5,1.0), "")
    plt.show()


def retrieveTree(i):
    listOfTrees = [{"no surfacing": {0: "no", 1: {"flippers": {0: "no", 1: "yes"}}}},
                   {"no surfacing": {0: "no", 1: {"flippers": {0: {"head": {0: "no", 1: "yes"}}, 1: "no"}}}}]
    return listOfTrees[i]


def getNumLeafs(myTree):
    """
    获取叶节点数目
    :param myTree: 树
    :return:  叶节点数目
    """
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in list(secondDict.keys()):
        if type(secondDict[key]).__name__ == "dict":
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    """
    获取树的层数
    :param myTree: 树
    :return: 树的层数
    """
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in list(secondDict.keys()):
        if type(secondDict[key]).__name__ == "dict":
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth

