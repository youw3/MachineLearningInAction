import numpy as np


def load_data(data_source):
    file = open(data_source, 'r')
    len_data = len(file.readlines())
    train_data = np.zeros((len_data, 3))
    label = []
    index = 0
    file = open(data_source, 'r')
    for line in file.readlines():
        line = line.strip()
        list_single_line = line.split('\t')
        train_data[index, :] = list_single_line[0:3]
        label.append(list_single_line[-1])
        index += 1
    return train_data, label


def classify_fn(k, test_data, train_data, labels):
    test_data = np.tile(test_data, [train_data.shape[0], 1])
    distance = (np.sum((test_data - train_data)**2, axis=1))**0.5
    sortedDistance = np.argsort(distance)
    dict_labelsCount={}
    for i in range(k):
        label = labels[sortedDistance[i]]
        dict_labelsCount[label] = dict_labelsCount.get(label, 0) + 1
    sortedDict_labelsCount = sorted(dict_labelsCount.items(), key=lambda d:d[1], reverse=True)
    return sortedDict_labelsCount[0][0]


def Norm(dataset):
    minVal = dataset.min(axis=0)
    maxVal = dataset.max(axis=0)
    ranges = maxVal-minVal
    m = dataset.shape[0]
    normDataset = dataset - np.tile(minVal, (m,1))
    normDataset = normDataset/np.tile(ranges, (m,1))
    return normDataset


if __name__ == '__main__':
    datingDataMat, datingLabels = load_data('datingTestSet2.txt')
    datingDataMat = Norm(datingDataMat)
    print(datingDataMat)
    test_ratio = 0.1
    m = datingDataMat.shape[0]
    number_test = int(m*test_ratio)
    error_count = 0.0
    for i in range(number_test):
        classifierResult = classify_fn(3, datingDataMat[i, :], datingDataMat[number_test:m, :], datingLabels
                                       [number_test:m])
        if classifierResult!=datingLabels[i]:
            error_count+=1
        print('predict:', classifierResult, 'real:', datingLabels[i])
    print('error rate:', error_count/number_test)