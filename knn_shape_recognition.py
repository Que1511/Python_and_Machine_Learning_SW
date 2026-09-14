# coding=utf-8
import os
from os import listdir

import numpy as np
from PIL import Image

IMG_SIZE = 16
K = 3

LABELS = {
    "circle": 0,
    "square": 1,
    "triangle": 2
}

LABEL_NAMES = {
    0: "圆形",
    1: "正方形",
    2: "三角形"
}


def img2vector(filename, h=16, w=16):
    """读取图片，预处理后转换成 1×(h*w) 的一维向量。"""
    img = Image.open(filename).convert("L")
    img = img.resize((w, h))

    pixels = np.array(img)
    binary_img = (pixels < 128).astype(int)

    imgVector = np.zeros((1, h * w))
    for row in range(h):
        for col in range(w):
            imgVector[0, row * w + col] = binary_img[row, col]

    return imgVector


def loadFolder(folder, label):
    """读取一个类别文件夹中的所有图片。"""
    fileList = sorted(listdir(folder))
    dataX = []
    dataY = []

    for fileName in fileList:
        if not fileName.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            continue

        filePath = os.path.join(folder, fileName)
        vector = img2vector(filePath, IMG_SIZE, IMG_SIZE)

        dataX.append(vector[0])
        dataY.append(label)

    return dataX, dataY


def loadDataSet():
    """加载训练集和测试集。"""
    print("1.Loading trainSet...")

    trainX = []
    trainY = []

    for className, label in LABELS.items():
        folder = os.path.join("shape_data", "trainSet", className)
        dataX, dataY = loadFolder(folder, label)
        trainX.extend(dataX)
        trainY.extend(dataY)

    print("2.Loading testSet...")

    testX = []
    testY = []

    for className, label in LABELS.items():
        folder = os.path.join("shape_data", "testSet", className)
        dataX, dataY = loadFolder(folder, label)
        testX.extend(dataX)
        testY.extend(dataY)

    return np.array(trainX), trainY, np.array(testX), testY


def myKNN(testData, trainX, trainY, k):
    """
    使用 KNN 对一个测试样本进行分类。

    参数：
        testData：当前待分类样本的一维向量
        trainX：训练集特征矩阵
        trainY：训练集标签
        k：参与投票的最近邻数量

    返回：
        预测类别标签：0、1 或 2

    请完成：
    1. 计算 testData 与所有训练样本之间的欧式距离
    2. 按距离从小到大排序
    3. 取距离最近的 k 个训练样本
    4. 统计这 k 个样本的类别票数
    5. 返回票数最多的类别
    """

    # 1. 计算 testData 与所有训练样本之间的欧式距离
    numSamples = trainX.shape[0]
    distance = np.zeros(numSamples)

    for i in range(numSamples):
        diff = testData - trainX[i]        # 逐维相减
        squaredDiff = diff ** 2            # 差值平方
        squaredDist = squaredDiff.sum()    # 所有维度求和
        distance[i] = np.sqrt(squaredDist)  # 开平方，得到欧式距离

    # 2. 按距离从小到大排序，得到训练样本下标
    sortedDistIndices = np.argsort(distance)

    # 3. 取距离最近的 k 个训练样本，统计类别票数
    classCount = {}
    for i in range(k):
        voteLabel = trainY[sortedDistIndices[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

    # 4. 返回票数最多的类别（平票时保留先出现的类别）
    maxCount = -1
    bestLabel = None
    for label, count in classCount.items():
        if count > maxCount:
            maxCount = count
            bestLabel = label

    return bestLabel


def main():
    trainX, trainY, testX, testY = loadDataSet()

    print("训练集大小：", trainX.shape)
    print("测试集大小：", testX.shape)

    print("3.Start KNN classification...")

    matchCount = 0
    numTestSamples = testX.shape[0]

    for i in range(numTestSamples):
        predict = myKNN(testX[i], trainX, trainY, K)

        print(
            "第%d张：预测=%s，真实=%s"
            % (i + 1, LABEL_NAMES[predict], LABEL_NAMES[testY[i]])
        )

        if predict == testY[i]:
            matchCount += 1

    accuracy = float(matchCount) / numTestSamples

    print("4.Show the result...")
    print("测试样本总数：", numTestSamples)
    print("识别错误数量：", numTestSamples - matchCount)
    print("分类准确率：%.2f%%" % (accuracy * 100))


if __name__ == "__main__":
    main()
