import numpy as np


def load_data_set():
    """
    创建数据集,都是假的 fake data set
    :return: 单词列表posting_list, 所属类别class_vec
    """
    posting_list = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'gar e'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0, 1, 0, 1, 0, 1]  # 1 is 侮辱性的文字, 0 is not
    return posting_list, class_vec


def createVocablist(dataset):
    vocabset = set([])
    for document in dataset:
        vocabset = set.union(vocabset, document)
    return list(vocabset)


def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pclass1 = np.sum(trainCategory) / numTrainDocs
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += np.sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += np.sum(trainMatrix[i])
    p1vect = np.log(p1Num/p1Denom)
    p0Vect = np.log(p0Num/p0Denom)
    return p0Vect, p1vect, pclass1


def classify_fn(testVec, p0Vec, p1Vec, pClass1):
    p1 = sum(testVec*p1Vec) + np.log(pClass1)
    p0 = sum(testVec*p0Vec) + np.log(1-pClass1)
    return p1 > p0


if __name__ == '__main__':
    listOfPosts, listClasses = load_data_set()
    vocabset = createVocablist(listOfPosts)
    trainMat = []
    for postinDoc in listOfPosts:
        trainMat.append(setOfWords2Vec(vocabset, postinDoc))

    p0V, p1V, pAb = trainNB0(trainMat, np.array(listClasses))
    print(p0V, p1V, pAb)
    testVec = ['love', 'my', 'dalmation']
    print(classify_fn(np.array(setOfWords2Vec(vocabset, testVec)), p0V, p1V, pAb))
    testVec1 = ['stupid', 'garbage']
    print(classify_fn(setOfWords2Vec(vocabset, testVec1), p0V, p1V, pAb))

