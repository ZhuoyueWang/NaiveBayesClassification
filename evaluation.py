import numpy as np
import math

class BayesClassification:

    def __init__(self):
        self.total = 0
        self.truePos = 0
        self.trueNeg = 0
        self.falsePos = 0
        self.falseNeg = 0
        self.numCorrect = 0
        self.numTest = 0
        self.numClass = 0
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
        loc = open(trainData,'r')
        data = loc.readlines()
        length = len(data)
        self.total = length
        width = len(data[0])
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
                    self.probTable[trainLabel[i]][j] += 1.0
            self.classSet[trainLabel[i]] += 1

        for i in range(self.numClass):
            for j in range(self.numFeature):
                self.probTable[i][j] = self.probTable[i][j]/self.classSet[i]
            self.priors.append(self.classSet[i]/self.total)


    def dataTest(self,testData):
        loc = open(trainData,'r')
        data = loc.readlines()
        length = len(data)
        self.numTest = length
        width = len(data[0])
        self.numFeature = width
        self.numCorrect = 0
        for i in range(self.numClass):
            feature = []
            for i in range(self.numFeature):
                feature.append(0)
            self.confidenceInter.append(feature)
            self.testClassSet.append(0)

        for i in range(length):
            pLabel = sampleTest(testData[i])
            self.probLabel.append(pLabel)
            self.confidenceInter[self.testLabel[i]][pLabel] += 1
            self.testClassSet[self.testLabel[i]] += 1

        for i in range(self.numClass):
            feature = []
            for j in range(self.numFeature):
                feature.append(self.confidenceInter[i][j]/self.testClassSet[i])
            self.confusionMatrix.append(feature)
            self.numCorrect += self.confidenceInter[i][i]


    def sampleTest(self,sample):
        maxID = 0
        localMax = -99999

        for i in range(numClass):
            p = math.log(self.priors[i])
            for j in range(numFeature):
                pp = self.probTable[i][j]
                if p < 0.00001:
                    p = 0.00001
                if p > 0.99999:
                    p = 0.99999
                if sample[j] == 1:
                    p += math.log(pp)
                if sample[j] == 0:
                    p -= math.log(pp)
            if localMax < p:
                localMax = p
                maxID = i
            self.posterior.append(p)
        return maxID



    def confusionMatrix(self,x,y):
        p1List = []
        p2List = []
        oddList = []
        print('High confusion rate to classify class {0} as class {1} is {2}'.format(x,y,self.confusionMatrix[x][y]))

        for i in range(numFeature):
            p1 = probTable[x][i]
            p2 = probTable[y][i]
            if p1 < 0.00001:
                p1 = 0.00001
            if p1 < 0.99999:
                p1 = 0.99999
            if p2 < 0.00001:
                p2 = 0.00001
            if p2 > 0.99999:
                p2 = 0.99999
            p1List.append(math.log(p1))
            p2List.append(math.log(p2))
            oddList.append(math.log(p1)-math.log(p2))

        print('log likelihood for class 1:')
        for i in range(numFeature):
            print(p1List[i], end= ' ')
        print()

        print('log likelihood for class 2:')
        for i in range(numFeature):
            print(p2List[i], end = ' ')
        print()

        print('log odd ratio')
        for i in range(numFeature):
            print(oddList[i], end = ' ')
        print()


    def printMatrix(self):
        print("confusion Matrix:")
        for i in range(numClass):
            for i in range(numFeature):
                print(confusionMatrix[i][j], end = ',')
            print()


    def printTable(self):
        print("Probability Table:")
        for i in range(numClass):
            for i in range(numFeature):
                print(probTable[i][j], end = ',')
            print()






def main():
    traindata =


if __name__ == "__main__":
    main()
