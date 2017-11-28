import numpy as np
import mpmath as math
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import deque


def read_data(data,label):
    trainingFile = open(data, "r")
    labelFile = open(label,"r")
    lines_data = trainingFile.readlines()
    lines_label = labelFile.readlines()
    alldata = [[],[],[],[],[]]
    for i in range(len(lines_label)):
        data = []
        label = lines_label[i].rstrip("\n")
        #print(label)
        for j in range(33*i,33*i+30):
            line_data = lines_data[j]

            line_data = line_data.rstrip('\n')
            elem = [i for i in line_data]
            data.append(elem)
        alldata[int(label)-1].append(data)

    #print(np.shape(alldata))
    alldata = pre_process(alldata)


    return alldata



def pre_process (data_set):
    [num,depth,rows, columns] = np.shape(data_set)
    processed = [[] for x in range(num)]
    #print(processed)
    for idx in range(num):
        for k in range(depth):
            matrix = [[0 for x in range(columns)] for y in range(rows)]
            data = data_set[idx][k]
            for i in range(rows):
                for j in range(columns):
                    if data[i][j] == '%':
                        matrix[i][j] = 1
            processed[idx].append(matrix)

    return processed

def part2_2_classifier(training_data,testing_data):
    [num,train_depth,train_rows,train_columns] = np.shape(training_data)
    # in this part, the numbers of training data happen to be the same for all the labels
    prior = [1/num for x in range(num)]

    prob_table = [[[0 for x in range(train_columns)] for y in range(train_rows)] for z in range(num)]
    #print(np.shape(prob_table))
    for idx in range(num):
        for depth in range(train_depth):
            data = training_data[idx][depth]
            for i in range(train_rows):
                for j in range(train_columns):
                    prob_table[idx][i][j] += data[i][j]


    # Laplace Smoothing
    k = 3
    V = 2
    for idx in range(num):
        for i in range(train_rows):
            for j in range(train_columns):
                prob_table[idx][i][j] += k
                prob_table[idx][i][j] = prob_table[idx][i][j]/(train_depth+k*V)

    decision_test = [[],[],[],[],[]]
    correctness = 0
    [num,test_depth,test_rows, test_columns] = np.shape(testing_data)
    Prob = [[0 for x in range(num)] for y in range(num)]  # 1st idx is the real labek, 2nd idx is the probability of how it is labelled
    # classify the testing data
    for idx in range(num):
        for depth in range(test_depth):
            data = testing_data[idx][depth]
            likelyhood = [0 for x in range(num)]
            for i in range(test_rows):
                for j in range(test_columns):
                    if data[i][j] == 1:
                        for k in range(num):
                            likelyhood[k] += math.log(prob_table[k][i][j])
            for k in range(num):
                Prob[idx][k] =  math.log(prior[k]) + likelyhood[k]
            label =  np.argmax(Prob[idx]) + 1
            decision_test[idx].append(label)
            if label == idx + 1:
                correctness += 1

    print("The labels of the testing data are")
    for i in range(len(decision_test)):
        print(decision_test[i])

    print("percentage of correctness for testing = " + str(correctness / (num*test_depth)))
    confusion = []
    for i in range(num):
        count = [0 for x in range(num)]
        for j in range(test_depth):
            label = decision_test[i][j]
            count[label - 1] += 1
        a = np.array(count)/test_depth
        confusion.append(a)
    print("the confusion matrix is ")
    # for i in range(len(confusion)):
    #     print(confusion[i])
    print(np.matrix(confusion))

def main():
    training_data = read_data("training_data.txt","training_labels.txt")
    testing_data = read_data("testing_data.txt", "testing_labels.txt")
    result = part2_2_classifier(training_data,testing_data)

if __name__== "__main__":
  main()
