import numpy as np
import mpmath as math
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import deque


def read_training_data1():
    trainingFile = open("trainingimages", "r")
    lines = trainingFile.readlines()
    image_num = int(len(lines)/28)
    image_data = []
    for i in range(image_num):
        data = []
        for j in range(28*i,28*i+28):
            line = lines[j]
            line = line.rstrip('\n')
            elem = [a for a in line]
            data.append(elem)
        image_data.append(data)
    print(np.shape(image_data))
    image_data = pre_process(image_data)

    trainingLabel = open("traininglabels", "r")
    labels = trainingLabel.readlines()
    data_depth = len(labels)
    data_labels = []
    for i in range(len(labels)):
        label = labels[i]
        label = label.rstrip('\n')
        data_labels.append(label)

    return image_data, data_labels, data_depth

def read_test_data1():
    testFile = open("testimages", "r")
    lines = testFile.readlines()
    image_num = int(len(lines)/28)
    image_test = []
    for i in range(image_num):
        data = []
        for j in range(28 * i, 28 * i + 28):
            line = lines[j]
            line = line.rstrip('\n')
            elem = [a for a in line]
            data.append(elem)
        image_test.append(data)
    print(np.shape(image_test))
    image_test = pre_process(image_test)

    testLabel = open("testlabels", "r")
    labels = testLabel.readlines()
    test_depth = len(labels)
    test_labels = []
    for i in range(labels):
        label = labels[i]
        label = label.rstrip('\n')
        test_labels.append(label)

    return image_test, test_labels, test_depth

def pre_process(data_set):
    print(np.shape(data_set))
    print("what")
    [depth,rows, columns] = np.shape(data_set)
    processed = []
    for k in range(depth):
        matrix = [[0 for x in range(columns)] for y in range(rows)]
        data = data_set[k]
        for i in range(rows):
            for j in range(columns):
                if data[i][j] == '%' or data[i][j] == '+':
                    matrix[i][j] = 1
        processed.append(matrix)
    return processed



def read_training_data():
    trainingFile = open("trainingimages", "r")
    lines = trainingFile.readlines()
    image_num = int(len(lines)/28)
    image_data = []
    for i in range(image_num):
        data = []
        for j in range(28*i,28*i+28):
            line = lines[j]
            line = line.rstrip('\n')
            elem = [a for a in line]
            data.append(elem)
        image_data.append(data)
    print(np.shape(image_data))
    image_data = pre_process(image_data)

    trainingLabel = open("traininglabels", "r")
    labels = trainingLabel.readlines()
    data_depth = len(labels)
    data_labels = []
    for i in range(len(labels)):
        label = labels[i]
        label = label.rstrip('\n')
        data_labels.append(label)

    return image_data, data_labels, data_depth    

def part1_1_classifier(image_data,data_labels,data_depth,image_test,test_labels,test_depth):
    [image_depth,image_rows, image_columns] = np.shape(image_data)

    priors = [0 for i in range(10)]
    for i in data_labels:
        priors[i] += 1
    for i in priors:
        i /= data_depth

    prob_table = [0 for i in range(10)]
    for i in prob_table:
        for x in range(image_rows):
            for y in range(image_columns):
                prob_table[i][x][y] = 0

    for i in range(image_depth):
        data = image_data[i]
        label = data_labels[i]
        for x in range(image_rows):
            for y in range(image_columns):
                prob_table[label][x][y] += data[x][y]

    # Laplace Smoothing
    k = 0.7
    V = 2
    for i in prob_table:
        for x in range(image_rows):
            for y in range(image_columns):
                prob_table[i][x][y] += k
                prob_table[i][x][y] = prob_table[i][x][y]/(data_depth+k*V)

    [test_depth,test_rows, test_columns] = np.shape(image_test)
    posterior = []

    for i in range(test_depth):
        data = image_test[i]
        localmax = -99999
        guessDigit = -1
        for a in range(10):
            likelyhood = 0
            for x in range(test_rows):
                for y in range(test_columns):
                    if data[x][y] == 1:
                        likelyhood += math.log(prob_table[a][x][y])
            P =  math.log(prior[a]) + likelyhood
            if P > localmax:
                localmax = P
                guessDigit = a
        posterior.append((guessDigit,localmax))


    confusion = [[0 for x in range(10)] for y in range(10)]
    #row是实际值 col是learn的结果
    highPosterior = [-9999 for i in range(10)]
    lowPosterior = [9999 for i in range(10)]
    for i in data_labels:
        for j in posterior:
            confusion[i][posterior[j][0]] += 1
            if i == posterior[j][0]:
                if posterior[j][1] > highPosterior[i]:
                    highPosterior[i] = posterior[j][1]
                if posterior[j][1] < lowPosterior[i]:
                    lowPosterior[i] = posterior[j][1]

    totalAccuracy = 0
    for i in range(10):
        for j in range(10):
            confusion[i][j] /= test_depth
            if i == j:
                totalAccuracy += confusion[i][j]
    totalAccuracy /= 10

    print("the confusion matrix is ")
    print(confusion)
    print("the total accuracy is ")
    print(totalAccuracy)

    print("odd ratio:")




def main():
    image_data, data_labels,data_depth = read_training_data()
    image_test, test_labels,test_depth = read_test_data()
    result = part1_1_classifier(image_data,data_labels,data_depth,image_test,test_labels,test_depth)

if __name__== "__main__":
  main()
