import numpy as np
from math import log


def calEntropy(dataset):
    lenofdata = len(dataset)
    labelCounts = {}
    for singledata in dataset:
        label = singledata[-1]
        if label not in labelCounts.keys():
            labelCounts[label] = 0
        labelCounts[label] += 1
    entropy=0
    for key in labelCounts.keys():
        prob = float(labelCounts[key]) / lenofdata
        entropy -= prob * log(prob, 2)
    return entropy


def createDataset():
    dataset = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'],
               [0, 1, 'no']]

    labels = ['no surfacing', 'flippers']
    return dataset, labels


def splitDataset(dataset, axis, value):
    retDataSet = []
    for singledata in dataset:
        if singledata[axis] == value:
            reduceddata = singledata[:axis]
            reduceddata.extend(singledata[axis+1:])
            retDataSet.append(reduceddata)
    return retDataSet


def chooseBestFeatureToSplit(dataset):
    baseEntropy = calEntropy(dataset)
    numFeatures = len(dataset[0]) - 1    # last is label
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):   # pick the best feature
        fealist = [example[i] for example in dataset]
        uniqueVals = set(fealist)  # get unique value of this feature
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataset(dataset, i, value)
            prob = len(subDataSet) / len(dataset)
            newEntropy += prob * calEntropy(subDataSet)
        infogain = baseEntropy - newEntropy
        if infogain>bestInfoGain:
            bestInfoGain = infogain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=lambda d: d[1], reverse=True)
    return sortedClassCount[0][0]


def createTree(dataset, labels):
    classList = [example[-1] for example in dataset]
    if classList.count(classList[0]) == len(dataset):  # the same label
        return classList[0]
    if len(dataset[0]) == 1:         # only a feature and can't split
        return majorityCnt(dataset)
    bestFeat = chooseBestFeatureToSplit(dataset)
    bestLabel = labels[bestFeat]
    myTree = {bestLabel: {}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataset]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestLabel][value] = createTree(splitDataset(dataset, bestFeat, value), subLabels)
    return myTree


def classify_fn(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify_fn(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


if __name__ == '__main__':
    myDat, labels = createDataset()
    tree = createTree(dataset=myDat, labels=labels)
    print(tree)
    myDat, labels = createDataset()
    print(classify_fn(tree, labels, [1, 1]))