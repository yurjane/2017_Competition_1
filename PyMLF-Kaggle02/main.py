import math
import operator
import csv

from pysnooper import snoop
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



class rootNode():
    def __init__(self, attr, value, lchild=None, rchild=None):
        self.attr = attr
        self.value = value
        self.lchild = lchild
        self.rchild = rchild


class leafNode():
    def __init__(self, label):
        self.label = label


class cartTree():
    def __init__(self, dataSet:pd.DataFrame):
        self.categoryMode = list()
        labels = sorted(list(set(dataSet['T'])))
        for label in labels:
            self.categoryMode.append(dataSet[dataSet['T'] == label].mean())
        self.mean = np.array(dataSet.mean()[:-1])
        self.std = np.array(dataSet.std()[:-1])
        dataSet = np.array(dataSet)
        dataSet[:, :-1] = (dataSet[:, :-1] - self.mean) / self.std
        # dataSet, self.reconMat = self.pca(dataSet[:, :-1], 4)
        self.root = self.createTree(dataSet, list(range(dataSet.shape[1] - 1)))

    # @snoop()
    def pca(self, dataMat, topNfeat=999999):
        covMat = np.cov(dataMat, rowvar=0)  # rowvar=0-->以列代表一个变量，计算各列之间的协方差
        eigVals, eigVects = np.linalg.eig(np.mat(covMat))  # 协方差矩阵的特征值和特征向量
        eigValInd = np.argsort(eigVals)
        eigValInd = eigValInd[:-(topNfeat + 1):-1]  # 对升序排序结果从后往前取topNfeat个值
        redEigVects = eigVects[:, eigValInd]  # 取选定特征值对应的特征向量，从而转换原始数据
        dataMat = dataMat * redEigVects  # 将原始数据转换到新空间
        return dataMat, redEigVects

    # @snoop()
    def createTree(self, dataSet, ableLabel:list):
        if len(ableLabel) != 0:
            bestAttr = self.choose_best_feature(np.asarray(dataSet), ableLabel)
            if bestAttr is None:
                return leafNode(self.argmaxlabel(dataSet))
            bestValue = self.choose_best_value(np.asarray(dataSet), bestAttr)
            if bestValue is None:
                return leafNode(self.argmaxlabel(dataSet))
            leftSet = dataSet[dataSet[:, bestAttr] <= bestValue]
            rightSet = dataSet[dataSet[:, bestAttr] > bestValue]
            root = rootNode(bestAttr, bestValue)
            new_labels = ableLabel.copy()
            new_labels.remove(bestAttr)
            if leftSet.shape[0] > 50:
                root.lchild = self.createTree(leftSet, new_labels)
            else:
                root.lchild = leafNode(self.argmaxlabel(leftSet))
            if rightSet.shape[0] > 50:
                root.rchild = self.createTree(rightSet, new_labels)
            else:
                root.rchild = leafNode(self.argmaxlabel(rightSet))
            return root
        return leafNode(self.argmaxlabel(dataSet))

    def get_shannon_entropy(self, dataSet):
        """
        获得结点的香农熵
        :param dataSet: 数据集
        :return: 香农熵
        """
        elemNum = len(dataSet)
        labelNums = dict()
        for elem in dataSet:
            elemLabel = elem[-1]
            if elemLabel not in labelNums:
                labelNums[elemLabel] = 0
            labelNums[elemLabel] += 1
        shannonEnt = 0.0
        for key in labelNums.keys():
            prob = float(labelNums[key])/elemNum
            shannonEnt -= prob * math.log(prob, 2)
        return shannonEnt

    def split_dataSet(self, dataSet, axis, value):
        spl_dataSet = list()
        for elem in dataSet:
            if elem[axis] == value:
                reducedFeatVec = elem[:axis]
                reducedFeatVec.extend(elem[axis + 1:-1])
                spl_dataSet.append(reducedFeatVec)
        return spl_dataSet

    def choose_best_feature(self, dataSet, ableLabel):
        attrNum = len(dataSet[0]) - 1
        baseEnt = self.get_shannon_entropy(dataSet)
        bestGain = -np.inf
        bestAttr = None
        for sub in ableLabel:
            featList = [elem[sub] for elem in dataSet]
            featSet = set(featList)
            newEnt = 0.0
            for value in featSet:
                aDatas = self.split_dataSet(dataSet.tolist(), sub, value)
                prob = len(aDatas) / float(len(dataSet))
                newEnt += prob * self.get_shannon_entropy(dataSet)
            infoGain = baseEnt - newEnt
            if infoGain > bestGain:
                bestGain = infoGain
                bestAttr = sub
        return bestAttr

    def choose_best_value(self, dataSet, bestAttr):
        values = dataSet[:, bestAttr]
        valueSet = sorted(list(set(values.tolist())))
        prepareValues = [(valueSet[sub] + valueSet[sub + 1]) / 2 for sub in range(len(valueSet) - 1)]
        baseEnt = self.get_shannon_entropy(dataSet)
        bestGain = -np.inf
        bestValue = None
        for prepareValue in prepareValues:
            smallSet = dataSet[dataSet[:, bestAttr] <= prepareValue]
            bigSet = dataSet[dataSet[:, bestAttr] > prepareValue]
            infoGain = baseEnt - (len(smallSet) * self.get_shannon_entropy(smallSet) + len(bigSet)
                                  * self.get_shannon_entropy(bigSet)) / len(dataSet)
            if infoGain > bestGain:
                bestGain = infoGain
                bestValue = prepareValue
        return bestValue

    def majorityCnt(self, classList):
        classCount = {}
        for vote in classList:
            if vote not in classCount.keys():
                classCount[vote] = 0
            classCount[vote] += 1
        return max(classCount)

    def argmaxlabel(self, dataSet):
        labelDict =dict()
        maxNum = 0
        bestLabel = None
        for elem in dataSet:
            if elem[-1] not in labelDict.keys():
                labelDict[elem[-1]] = 0
            labelDict[elem[-1]] +=1
        for key, value in labelDict.items():
            if value > maxNum:
                maxNum = value
                bestLabel = key
        return bestLabel

    def search(self, elem):
        dealElem = ((elem - self.mean) / self.std).tolist()
        labels = self.sampls_search(self.root, dealElem)
        labels = list(set(labels))
        if len(labels) == 1:
            return labels[0]
        return self.nearset_search(elem, labels)

    def sampls_search(self, root:rootNode or leafNode, elem):
        if isinstance(root, leafNode):
            return [root.label]
        labels = list()
        if elem[root.attr] > root.value:
            labels.extend(self.sampls_search(root.rchild, elem))
        elif elem[root.attr] < root.value:
            labels.extend(self.sampls_search(root.lchild, elem))
        else:
            labels.extend(self.sampls_search(root.lchild, elem))
            labels.extend(self.sampls_search(root.rchild, elem))
        return labels

    def nearset_search(self, elem, labels):
        distanceSet = list()
        for label in labels:
            distanceSet.append(self.euclidean_distance(elem, self.categoryMode[int(label) - 1]))
        minDis = np.inf
        bestLabel = None
        for sub in range(len(distanceSet)):
            if distanceSet[sub] < minDis:
                minDis = distanceSet[sub]
                bestLabel = labels[sub]
        return bestLabel

    def euclidean_distance(self, var1, var2):
        distance = 0
        for value1, value2 in zip(var1, var2):
            if math.isnan(value1):
                continue
            distance += (value1 - value2) ** 2
        return math.sqrt(distance)



dataset = pd.read_csv("./train.csv")
dataset = dataset.replace('?', np.nan)
dataset = dataset.astype(float)

labels = set(dataset['T'])
for label in labels:
    dataset[dataset['T'] == label] = dataset[dataset['T'] == label].fillna(dataset[dataset['T'] == label].mean())
    # dataset[dataset['T'] == label].boxplot()
    # plt.title(f'label={label}')
    # plt.show()
testData = pd.read_csv("./train.csv")
testData = testData.replace('?', np.nan)
testData = testData.astype(float)
del testData['T']

labels = dataset['T']
tree = cartTree(dataset)
#
pro_labels = list()
for elem in np.array(testData):
    label = tree.search(elem)
    pro_labels.append(label)
pro_labels = np.array(pro_labels, dtype=int)
# print(pro_labels)
labels = np.array(labels, dtype=int)
pro_labels = np.array(pro_labels, dtype=int)
print(pro_labels)
print((labels == pro_labels).tolist().count(True))
# # pd.DataFrame(pro_labels).to_csv('./pro.csv')
# print(type(np.array(testData)[0, 9]))