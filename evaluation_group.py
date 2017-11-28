import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import deque
import pprint


def read_training_data():
    trainingFile = open("trainIamageOutput1.txt", "r")
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
    testFile = open("testIamageOutput1.txt", "r")
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



def part1_2_classifier_disjoint(image_data,data_labels,data_depth,image_test,test_labels,test_depth, length, width):
    [image_depth,image_rows, image_columns] = np.shape(image_data)

    priors = [0 for i in range(10)]
    for i in data_labels:
        priors[i] += 1
    for i in range(len(priors)):
        priors[i] /= data_depth

    num_features = math.pow(10, length*width)
    num_features = int(num_features)
    prob_table = [[[[0 for f in range(num_features)] for k in range(image_columns)] for j in range(image_rows)] for i in range(10)]
    totals = []
    for i in range(data_depth):
        data = image_data[i]
        label = data_labels[i]
        for x in range(0, image_rows-length, length):
            for y in range(0, image_columns-width, width):
                value = 1;
                total = 0;
                for l in range(length):
                    for w in range(width):
                        #print("value")
                        if data[x+l][y+w] == 1:
                            total += value;
                        value *= 10;
                totals.append(total)
                prob_table[label][x][y][total] += 1

    # Laplace Smoothing

    k = 0.01
    V = 2
    for i in range(len(prob_table)):
        digit_total = data_labels.count(i)+k*V
        for x in range(0, image_rows-length, length):
            for y in range(0, image_columns-width, width):
                for f in totals:
                        print(f)
                        prob_table[i][x][y][f] += k
                        prob_table[i][x][y][f] = prob_table[i][x][y][f]/digit_total


    [test_depth,test_rows, test_columns] = np.shape(image_test)
    posterior = []

    for i in range(test_depth):
        #print("test here")
        data = image_test[i]
        data1 = image_data[i]
        localmax = -99999
        guessDigit = -1
        for a in range(10):
            likelyhood = 0
            for x in range(0, test_rows-length, length):
                for y in range(0, test_columns-width, width):
                    value = 1;
                    total = 0;
                    for l in range(length):
                        for w in range(width):
                            if data[x+l][y+w] == 1:
                                total += value;
                            value *= 10;
                    likelyhood += math.log(prob_table[a][x][y][total])
            P = math.log(priors[a]) + likelyhood
            if P >= localmax:
                localmax = P
                guessDigit = a
        posterior.append((guessDigit,localmax))


    confusion = [[0 for x in range(10)] for y in range(10)]
    #row是实际值 col是learn的结果

    highPosterior = [-9999 for i in range(10)]
    highPosteriorIndex = [-9999 for i in range(10)]
    lowPosterior = [9999 for i in range(10)]
    lowPosteriorIndex = [9999 for i in range(10)]
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



def main():
    image_data, data_labels,data_depth = read_training_data()
    image_test, test_labels,test_depth = read_test_data()
    result = part1_2_classifier_disjoint(image_data,data_labels,data_depth,image_test,test_labels,test_depth, 2, 2)

if __name__== "__main__":
  main()
