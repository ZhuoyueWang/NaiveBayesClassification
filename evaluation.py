import numpy as np


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
            for j in range(self.numFeature):
                self.probTable[i][j] = self.probTable[i][j]/self.classSet[i]
            self.priors.append(self.classSet[i]/self.total)            


    def sampleTest(self,sample):




    def confusionMatrix(self,x,y):



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
