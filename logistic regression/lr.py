import numpy as np


def load_data():
    datamat = []
    labelmat = []
    file = open('testSet.txt')
    for line in file.readlines():
        lineArr = line.strip().split()
        datamat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelmat.append(int(lineArr[2]))
    return datamat, labelmat


def sigmoid(inx):
    return 1.0/(1+np.exp(-inx))


def gradAscent(datamat, classlabels):
    datamat = np.array(datamat)
    m, n = datamat.shape
    labelmat = np.array(classlabels).transpose().reshape([m, 1])
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(np.matmul(datamat, weights))
        error = labelmat - h
        weights = weights + alpha * np.matmul(datamat.transpose(), error)
    return weights


def stocGradAscent0(dataMatrix, classLabels):
    datamat = np.array(dataMatrix)
    m, n = datamat.shape
    alpha = 0.001
    weights = np.ones((n, 1))
    for i in range(m):
        h = sigmoid(np.matmul(datamat[i], weights))
        error = classLabels[i] - h
        weights = weights + alpha * np.matmul(np.array(datamat[i]).transpose().reshape((3, 1)),
                                              np.array(error).reshape(1, 1))
    return weights


if __name__ == '__main__':
    datamat, labelmat = load_data()
    print(gradAscent(datamat, labelmat))
    print(stocGradAscent0(datamat, labelmat))