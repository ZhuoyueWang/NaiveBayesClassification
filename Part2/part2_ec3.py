import numpy as np
import mpmath as math
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import deque
import time

def read_training_data():
    trainingFile = open("yes_train.txt", "r")
    lines = trainingFile.readlines()
    yes_num = int(len(lines)/28)
    yes_data = []
    for i in range(yes_num):
        data = []
        for j in range(28*i,28*i+25):
            line = lines[j]
            line = line.rstrip('\n')
            elem = [i for i in line]
            data.append(elem)
        yes_data.append(data)
    yes_data = pre_process(yes_data)
    print(np.shape(yes_data))

    trainingFile = open("no_train.txt", "r")
    lines = trainingFile.readlines()
    no_num = int(len(lines) / 28)
    no_data = []
    for i in range(no_num):
        data = []
        for j in range(28 * i, 28 * i + 25):
            line = lines[j]
            line = line.rstrip('\n')
            elem = [i for i in line]
            data.append(elem)
        no_data.append(data)
    no_data = pre_process(no_data)
    print(np.shape(no_data))

    return yes_data,no_data

def read_test_data():

    testFile = open("yes_test.txt", "r")
    lines = testFile.readlines()
    yes_num = int(len(lines) / 28)
    yes_test = []
    for i in range(yes_num):
        data = []
        for j in range(28 * i, 28 * i + 25):
            line = lines[j]
            line = line.rstrip('\n')
            elem = [i for i in line]
            data.append(elem)
        yes_test.append(data)
    yes_test = pre_process(yes_test)
    print(np.shape(yes_test))

    testFile = open("no_test.txt", "r")
    lines = testFile.readlines()
    no_num = int(len(lines) / 28)
    no_test = []
    for i in range(no_num):
        data = []
        for j in range(28 * i, 28 * i + 25):
            line = lines[j]
            line = line.rstrip('\n')
            elem = [i for i in line]
            data.append(elem)
        no_test.append(data)
    no_test = pre_process(no_test)
    print(np.shape(no_test))

    return yes_test,no_test

def pre_process (data_set):
    [depth,rows, columns] = np.shape(data_set)
    processed = []
    for k in range(depth):
        matrix = [0 for y in range(rows)]
        data = data_set[k]
        for i in range(rows):
            count = 0
            for j in range(columns):
                if data[i][j] == '%':
                    count += 1
            matrix[i] = count/ columns
        processed.append(matrix)

    return processed

def part2_ec3_classifier(yes_data,no_data,yes_test,no_test):
    [yes_depth,yes_rows] = np.shape(yes_data)
    [no_depth,no_rows] = np.shape(no_data)
    prior_yes = yes_depth/(yes_depth+no_depth)
    prior_no = 1-prior_yes
    prob_table_yes = [[0 for x in range(11)] for y in range(yes_rows)]
    prob_table_no = [[0 for x in range(11)] for y in range(no_rows)]

    for depth in range(yes_depth):
        data = yes_data[depth]
        for i in range(yes_rows):
            ave_val = 10*data[i]
            prob_table_yes[i][int(ave_val)] += 1

    for depth in range(no_depth):
        data = no_data[depth]
        for i in range(no_rows):
            ave_val = 10 * data[i]
            prob_table_no[i][int(ave_val)] += 1

    # Laplace Smoothing
    k = 3
    V = 11
    for i in range(yes_rows):
        for j in range(11):
            prob_table_yes[i][j] += k
            prob_table_yes[i][j] = prob_table_yes[i][j]/(yes_depth+k*V)
            prob_table_no[i][j] += k
            prob_table_no[i][j] = prob_table_no[i][j]/(no_depth+k*V)

    decision_yes_test = []
    yes_right_count = 0
    yes_wrong_count = 0
    decision_no_test = []
    no_right_count = 0
    no_wrong_count = 0
    [yes_test_depth,yes_test_rows] = np.shape(yes_test)
    [no_test_depth,no_test_rows] = np.shape(no_test)

    # classify the yes_test data
    for depth in range(yes_test_depth):
        data = yes_test[depth]
        likelyhood_yes = 0
        likelyhood_no = 0
        for i in range(yes_test_rows):
            ave_val = data[i]*10
            likelyhood_yes += math.log(prob_table_yes[i][int(ave_val)])
            likelyhood_no += math.log(prob_table_no[i][int(ave_val)])

        P_yes =  math.log(prior_yes) + likelyhood_yes
        P_no = math.log(prior_no) + likelyhood_no
        if P_yes>P_no:
            decision_yes_test.append(1)
            yes_right_count += 1
        else:
            decision_yes_test.append(0)
            yes_wrong_count += 1

    # classify the no_test data
    for depth in range(no_test_depth):
        data = no_test[depth]
        likelyhood_yes = 0
        likelyhood_no = 0
        for i in range(no_test_rows):
            ave_val = data[i] *10
            likelyhood_yes += math.log(prob_table_yes[i][int(ave_val)])
            likelyhood_no += math.log(prob_table_no[i][int(ave_val)])
        P_yes = math.log(prior_yes) + likelyhood_yes
        P_no = math.log(prior_no) + likelyhood_no
        if P_yes > P_no:
            decision_no_test.append(1)
            no_wrong_count += 1

        else:
            decision_no_test.append(0)
            no_right_count += 1
    print(decision_yes_test)
    print(decision_no_test)
    print("percentage of correctness for yes_test = " + str(yes_right_count / yes_test_depth))
    print("percentage of correctness for no_test = " + str(no_right_count / no_test_depth))
    print("the overall correctness = " + str((yes_right_count+no_right_count) / (yes_test_depth+no_test_depth)))

    confusion = [[0 for x in range(2)] for y in range(2)]
    confusion[0][0] = yes_right_count / yes_test_depth
    confusion[0][1] = yes_wrong_count / yes_test_depth
    confusion[1][0] = no_wrong_count / no_test_depth
    confusion[1][1] = no_right_count / no_test_depth
    print("the confusion matrix is ")

    print("yes   " + str(confusion[0]))
    print("no    " + str(confusion[1]))


def main():
    yes_data,no_data = read_training_data()
    yes_test, no_test = read_test_data()
    #print(yes_data,no_data)
    #print(yes_test,no_test)
    start = time.time()

    result = part2_ec3_classifier(yes_data,no_data,yes_test,no_test)
    end = time.time()
    print("time taken: " ,str(end - start))
if __name__== "__main__":
  main()
