import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import deque
import pprint


def read_training_data():
    trainingFile = open("trainIamageOutput2.txt", "r")
    lines = trainingFile.readlines()
    image_num = int(len(lines)/28)
    image_data = []
    for i in range(image_num):
        data = []
        for j in range(28*i,28*i+28):
            line = lines[j]
            line = line.rstrip('\n')
            elem = [int(a) for a in line]
            data.append(elem)
        image_data.append(data)

    trainingLabel = open("traininglabels", "r")
    labels = trainingLabel.readlines()
    data_depth = len(labels)
    data_labels = []
    for i in range(len(labels)):
        label = labels[i]
        label = label.rstrip('\n')
        data_labels.append(int(label))

    return image_data, data_labels, data_depth



def read_test_data():
    testFile = open("testIamageOutput2.txt", "r")
    lines = testFile.readlines()
    image_num = int(len(lines)/28)
    image_test = []
    for i in range(image_num):
        data = []
        for j in range(28 * i, 28 * i + 28):
            line = lines[j]
            line = line.rstrip('\n')
            elem = [int(a) for a in line]
            data.append(elem)
        image_test.append(data)

    testLabel = open("testlabels", "r")
    labels = testLabel.readlines()
    test_depth = len(labels)
    test_labels = []
    for i in range(len(labels)):
        label = labels[i]
        label = label.rstrip('\n')
        test_labels.append(int(label))

    return image_test, test_labels, test_depth



def ternary_classifier(image_data,data_labels,data_depth,image_test,test_labels,test_depth):
    [image_depth,image_rows, image_columns] = np.shape(image_data)

    priors = [0 for i in range(10)]
    for i in data_labels:
        priors[i] += 1
    for i in range(len(priors)):
        priors[i] /= data_depth


    prob_table1 = [[[0 for k in range(image_columns)] for j in range(image_rows)] for i in range(10)]
    prob_table0 = [[[0 for k in range(image_columns)] for j in range(image_rows)] for i in range(10)]
    prob_table2 = [[[0 for k in range(image_columns)] for j in range(image_rows)] for i in range(10)]

    for i in range(data_depth):
        data = image_data[i]
        label = data_labels[i]
        for x in range(image_rows):
            for y in range(image_columns):
                if data[x][y] == 1:
                    prob_table1[label][x][y] += 1
                elif data[x][y] == 0:
                    prob_table0[label][x][y] += 1
                else:
                    prob_table2[label][x][y] += 1


    # Laplace Smoothing
    k = 0.1
    V = 3
    for i in range(len(prob_table0)):
        for x in range(image_rows):
            for y in range(image_columns):
                prob_table1[i][x][y] += k
                prob_table1[i][x][y] = prob_table1[i][x][y]/(data_labels.count(i)+k*V)
                prob_table0[i][x][y] += k
                prob_table0[i][x][y] = prob_table0[i][x][y]/(data_labels.count(i)+k*V)
                prob_table2[i][x][y] += k
                prob_table2[i][x][y] = prob_table2[i][x][y]/(data_labels.count(i)+k*V)


    [test_depth,test_rows, test_columns] = np.shape(image_test)
    posterior = []

    for i in range(test_depth):
        data = image_test[i]
        data1 = image_data[i]
        localmax = -99999
        guessDigit = -1
        for a in range(10):
            likelyhood = 0
            for x in range(test_rows):
                for y in range(test_columns):
                    if data[x][y] == 1:
                        likelyhood += math.log(prob_table1[a][x][y])
                    elif data[x][y] == 0:
                        likelyhood += math.log(prob_table0[a][x][y])
                    else:
                        likelyhood += math.log(prob_table2[a][x][y])
            P = math.log(priors[a]) + likelyhood
            if P >= localmax:
                localmax = P
                guessDigit = a
        posterior.append((guessDigit,localmax))


    confusion = [[0 for x in range(10)] for y in range(10)]

    highPosterior = [-999999 for i in range(10)]
    highPosteriorIndex = [-1 for i in range(10)]
    lowPosterior = [999999 for i in range(10)]
    lowPosteriorIndex = [-1 for i in range(10)]
    for i in range(len(posterior)):
        if posterior[i][1] > highPosterior[(posterior[i][0])] and posterior[i][0] == test_labels[i]:
            highPosterior[(posterior[i][0])] = posterior[i][1]
            highPosteriorIndex[(posterior[i][0])] = i
        if posterior[i][1] < lowPosterior[(posterior[i][0])] and posterior[i][0] == test_labels[i]:
            lowPosterior[(posterior[i][0])] = posterior[i][1]
            lowPosteriorIndex[(posterior[i][0])] = i

    for i in range(len(test_labels)):
        confusion[test_labels[i]][posterior[i][0]] += 1
    totalAccuracy = 0
    for i in range(10):
        for j in range(10):
            confusion[i][j] = round(confusion[i][j]/test_labels.count(i),4)
            if i == j:
                totalAccuracy += confusion[i][j]
    totalAccuracy = round(totalAccuracy/10,4)


    print("classification rates:")
    for i in range(10):
        print("{0}: {1}".format(i,confusion[i][i]))

    print("the confusion matrix is ")
    for i in range(10):
        for j in range(10):
            print(confusion[i][j], end=" ")
        print()
    print("the total accuracy is ")
    print(totalAccuracy)

    print("each digit's examples on the highest and lowest posterior probabilities")
    for i in range(10):
        print()
        print("digit: {}".format(i))
        print("highest posterior example")
        for a in range(28):
            for b in range(28):
                print(image_test[(highPosteriorIndex[i])][a][b], end='')
            print()
        print()
        print("lowest posterior example")
        for a in range(28):
            for b in range(28):
                print(image_test[(lowPosteriorIndex[i])][a][b], end='')
            print()


def main():
    image_data, data_labels,data_depth = read_training_data()
    image_test, test_labels,test_depth = read_test_data()
    result = ternary_classifier(image_data,data_labels,data_depth,image_test,test_labels,test_depth)

if __name__== "__main__":
  main()
