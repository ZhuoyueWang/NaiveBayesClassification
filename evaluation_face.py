import numpy as np
import math

class FaceBayesClassification:

    def __init__(self):
        self.total = 0
        self.numCorrect = 0
        self.numTest = 0
        self.numClass = 10
        self.numFeature = 0
        self.trainSet = []
        self.testSet = []
        self.probTable = []
        self.trainLabel = []
        self.testLabel = []
        self.probLabel = []
        self.classSet = []
        self.testClassSet = []
        self.confusionMatrix = []
        self.priors = []
        self.posterior = []
        self.confidenceInter = []
        self.pcp = []
        self.ncp = []


    def train(self,trainData,trainLabel):
        length = len(trainData)
        self.total = length
        width = len(trainData[0])
        self.numFeature = width
        print('There are {} features in this dataset'.format(width))

        for i in range(self.numClass):
            feature = []
            for i in range(self.numFeature):
                feature.append(0)
            self.probTable.append(feature)
            self.classSet.append(0)

        for i in range(length):
            for j in range(width):
                if trainData[i][j] == 1:
                    self.probTable[(trainLabel[i])][j] += 1.0
            self.classSet[trainLabel[i]] += 1

        for i in range(self.numClass):
            for j in range(self.numFeature):
                self.probTable[i][j] = self.probTable[i][j]/self.classSet[i]
            self.priors.append(self.classSet[i]/self.total)

    def sampleTest(self,sample):
        maxID = 0
        localMax = -9999999

        for i in range(self.numClass):
            p = math.log(self.priors[i])
            for j in range(self.numFeature):
                pp = self.probTable[i][j]
                if pp < 0.00001:
                    pp = 0.00001
                if pp > 0.99999:
                    pp = 0.99999
                if sample[j] == 1:
                    p += math.log(pp)
                if sample[j] == 0:
                    p += math.log(1-pp)
            if localMax < p:
                localMax = p
                maxID = i
            self.posterior.append(p)
        return maxID

    def dataTest(self,testData):
        for i in range(self.numClass):
            feature = []
            for j in range(self.numClass):
                feature.append(0)
            self.confidenceInter.append(feature)
            self.testClassSet.append(0)

        length = len(testData)
        self.numTest = length
        width = len(testData[0])
        self.numCorrect = 0

        for i in range(length):
            pLabel = self.sampleTest(testData[i])
            self.probLabel.append(pLabel)
            self.confidenceInter[self.testLabel[i]][pLabel] += 1
            self.testClassSet[self.testLabel[i]] += 1

        print("confidence")
        print(self.confidenceInter)
        for i in range(self.numClass):
            feature = []
            for j in range(self.numClass):
                feature.append(self.confidenceInter[i][j]/self.testClassSet[i])
            self.confusionMatrix.append(feature)
            self.numCorrect += self.confidenceInter[i][i]




    def getconfusionMatrix(self,x,y):
        p1List = []
        p2List = []
        oddList = []
        print('High confusion rate to classify class {0} as class {1} is {2}'.format(x,y,round(self.confusionMatrix[x][y],2)))

        for i in range(self.numFeature):
            p1 = self.probTable[x][i]
            p2 = self.probTable[y][i]
            if p1 < 0.00001:
                p1 = 0.00001
            elif p1 > 0.99999:
                p1 = 0.99999
            if p2 < 0.00001:
                p2 = 0.00001
            elif p2 > 0.99999:
                p2 = 0.99999
            p1List.append(math.log(p1))
            p2List.append(math.log(p2))
            oddList.append(math.log(p1)-math.log(p2))

        print('log likelihood for class 1:')
        for i in range(self.numFeature):
            print(round(p1List[i],2), end= ' ')
        print()

        print('log likelihood for class 2:')
        for i in range(self.numFeature):
            print(round(p2List[i],2), end = ' ')
        print()

        print('log odd ratio')
        for i in range(self.numFeature):
            print(round(oddList[i],2), end = ' ')
        print()


    def printMatrix(self):
        print("Confusion Matrix:")
        for i in range(self.numClass):
            for j in range(self.numClass):
                print(round(self.confusionMatrix[i][j],2), end = ',')
            print()


    def printTable(self):
        print("Probability Table:")
        for i in range(self.numClass):
            for j in range(self.numFeature):
                print(round(self.probTable[i][j],2), end = ',')
            print()




def main():
    digit = DigitBayesClassification()
    loc = open('trainIamageOutput1.txt','r')
    traindatas = loc.readlines()
    loc.close()
    loc = open('traininglabels','r')
    trainlabels = loc.readlines()
    loc.close()
    loc = open('testIamageOutput1.txt','r')
    testdatas = loc.readlines()
    loc.close()
    loc = open('testlabels','r')
    testlabels = loc.readlines()
    loc.close()

    for i in range(len(trainlabels)):
        traindata1 = traindatas[i].strip()
        trainlabel1 = trainlabels[i].strip()
        traindata = [int(i) for i in traindata1]
        trainlabel = int(trainlabel1)
        digit.trainLabel.append(trainlabel)
        digit.trainSet.append(traindata)

    for i in range(len(testlabels)):
        testdata1 = testdatas[i].strip()
        testlabel1 = testlabels[i].strip()
        testdata = [int(i) for i in testdata1]
        testlabel = int(testlabel1)
        digit.testLabel.append(testlabel)
        digit.testSet.append(testdata)

    digit.train(digit.trainSet,digit.trainLabel)
    digit.dataTest(digit.testSet)
    digit.printMatrix()
    print()
    digit.printTable()

    print('total correct numbers: {}, with odd ratio: {}'.format(digit.numCorrect,digit.numCorrect/digit.numTest))

    print('\n')
    digit.getconfusionMatrix(3,2)
    '''
    print('\n')
    digit.getconfusionMatrix(7,9)
    print('\n')
    digit.getconfusionMatrix(4,9)
    print('\n')
    digit.getconfusionMatrix(5,3)
    print('\n')
    '''

if __name__ == "__main__":
    main()
