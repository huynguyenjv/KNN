import csv
import numpy as numpy
import math

def loadDataset(path):
    file = open(path, "r")
    dataset = csv.reader(file)
    dataset = numpy.array(list(dataset))
    dataset = numpy.delete(dataset,0,0)
    numpy.random.shuffle(dataset)
    file.close()
    trainSet = dataset[:50]
    testSet = dataset[50:]

    return trainSet,testSet


def distance(pointA,pointB):
    feature = len(pointA) - 2;
    tmp = 0
    for i in range(feature):
        tmp += (float(pointA[i+1]) - float(pointB[i+1])) ** 2
    return math.sqrt(tmp)


def kNearestNeighbor(trainSet, point, k):
    distances = []
    for item in trainSet:
        distances.append({
            "label": item[-1],
            "value": distance(item, point)
        })
    distances.sort(key=lambda x: x["value"])
    labels = [item["label"] for item in distances]
    return labels[:k]


def findMostOccur(arr):
    labels = set(arr)
    ans = ""
    maxOccur = 0
    for label in labels:
        num = arr.count(label)
        if num > maxOccur:
            maxOccur = num
            ans = label
    return ans


if __name__ == "__main__":
    trainSet, testSet = loadDataset("./csv2.csv")
    numOfRightAnwser = 0
    for item in testSet:
        knn = kNearestNeighbor(trainSet, item, 10)
        answer = findMostOccur(knn)
        numOfRightAnwser += item[-1] == answer
        print("label: {} -> predicted: {}".format(item[0], answer))
    print("Accuracy", numOfRightAnwser / len(testSet))