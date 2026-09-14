# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt

from knn_shape_recognition import loadDataSet, myKNN, LABEL_NAMES


def create_folds(trainY, n_folds=5):
    """将训练集手动划分为 n_folds 折。"""
    folds = [[] for _ in range(n_folds)]
    labels = sorted(set(trainY))

    for label in labels:
        class_indices = []

        for i in range(len(trainY)):
            if trainY[i] == label:
                class_indices.append(i)

        for i, index in enumerate(class_indices):
            fold_index = i % n_folds
            folds[fold_index].append(index)

    return folds


def cross_validation(trainX, trainY, k_list, n_folds=5):
    """
    使用 5 折交叉验证，从多个候选 K 值中选择最佳 K。

    参数：
        trainX：训练集特征矩阵
        trainY：训练集标签
        k_list：待比较的 K 值列表，例如 [1, 3, 5, 7, 9]
        n_folds：交叉验证折数，默认 5

    返回：
        best_k：5 折平均准确率最高的 K 值
        k_scores：字典，保存每个 K 对应的平均准确率

    请完成：
    1. 调用 create_folds() 得到 5 折下标
    2. 遍历每个 K
    3. 对每个 K 进行 5 轮训练/验证
    4. 每轮 1 折验证，其余 4 折训练
    5. 使用 myKNN() 对验证集逐个分类
    6. 计算每一折准确率
    7. 计算当前 K 的 5 折平均准确率
    8. 保存到 k_scores
    9. 找到平均准确率最高的 best_k
    """

    # 1. 调用 create_folds() 得到 5 折样本下标
    folds = create_folds(trainY, n_folds)

    k_scores = {}

    # 2. 遍历每个 K
    for k in k_list:
        fold_scores = []

        # 3. 对当前 K 进行 5 轮训练/验证
        for i in range(n_folds):
            # 4. 每轮 1 折验证，其余 4 折训练
            val_indices = folds[i]
            train_indices = []

            for j in range(n_folds):
                if j != i:
                    train_indices.extend(folds[j])

            # 根据下标取出当前轮的训练集和验证集
            cv_trainX = trainX[train_indices]
            cv_trainY = [trainY[idx] for idx in train_indices]
            cv_valX = trainX[val_indices]
            cv_valY = [trainY[idx] for idx in val_indices]

            # 5. 使用 myKNN() 对验证集逐个分类
            correctCount = 0
            for idx in range(len(cv_valX)):
                predict = myKNN(cv_valX[idx], cv_trainX, cv_trainY, k)
                if predict == cv_valY[idx]:
                    correctCount += 1

            # 6. 计算当前折准确率
            fold_accuracy = float(correctCount) / len(cv_valX)
            fold_scores.append(fold_accuracy)

        # 7. 计算当前 K 的 5 折平均准确率，保存到 k_scores
        mean_score = float(sum(fold_scores)) / n_folds
        k_scores[k] = mean_score

    # 8. 找到平均准确率最高的 best_k（得分相同保留先出现的 K）
    best_k = k_list[0]
    best_score = k_scores[best_k]

    for k in k_list:
        if k_scores[k] > best_score:
            best_score = k_scores[k]
            best_k = k

    return best_k, k_scores


def predict_test_set(testX, trainX, trainY, k):
    """使用最佳 K 对测试集逐个预测。"""
    predictions = []

    for i in range(len(testX)):
        predict = myKNN(testX[i], trainX, trainY, k)
        predictions.append(predict)

    return predictions


def build_confusion_matrix(real_labels, predict_labels, num_classes):
    """
    手动构造三分类混淆矩阵。
    行：真实类别
    列：预测类别
    """
    confusion = np.zeros((num_classes, num_classes), dtype=int)

    for real, predict in zip(real_labels, predict_labels):
        confusion[real][predict] += 1

    return confusion


def calculate_metrics(confusion):
    """
    根据混淆矩阵手动计算模型评价指标。

    参数：
        confusion：三分类混淆矩阵，行表示真实类别，列表示预测类别

    返回：
        error_rate：模型整体错误率
        accuracy：模型整体准确率
        class_metrics：字典，保存每个类别的
                       TP、FP、FN、TN、
                       Precision、Recall、TPR、FPR、F1

    对每一个类别分别采用 One-vs-Rest 的方式，分别评价每个类别：
    1. 计算整体样本数 total
    2. 计算预测正确数量 correct
    3. 计算整体错误率和准确率
    4. 对每一个类别分别计算 TP、FP、FN、TN
    5. 计算 Precision
    6. 计算 Recall
    7. 计算 TPR
    8. 计算 FPR
    9. 计算 F1
    10. 将当前类别的所有指标保存到 class_metrics
    """

    num_classes = confusion.shape[0]

    # 1. 计算整体样本数 total
    total = int(confusion.sum())

    # 2. 计算预测正确数量 correct（主对角线之和）
    correct = 0
    for i in range(num_classes):
        correct += confusion[i][i]

    # 3. 计算整体错误率和准确率
    error_rate = float(total - correct) / total
    accuracy = float(correct) / total

    class_metrics = {}

    # 4. 对每一个类别分别采用 One-vs-Rest 方式评价
    for class_id in range(num_classes):
        # 当前类别视为正类，其余类别视为负类
        TP = int(confusion[class_id][class_id])
        FN = int(confusion[class_id].sum()) - TP          # 真实是当前类，但预测成其他类
        FP = int(confusion[:, class_id].sum()) - TP       # 真实是其他类，但预测成当前类
        TN = total - TP - FP - FN                         # 真实与预测都不是当前类

        # 5. Precision：预测为当前类的样本中，真正属于当前类的比例
        if TP + FP == 0:
            precision = 0.0
        else:
            precision = float(TP) / (TP + FP)

        # 6. Recall 与 7. TPR：真实属于当前类的样本中，被正确找出的比例
        if TP + FN == 0:
            recall = 0.0
            tpr = 0.0
        else:
            recall = float(TP) / (TP + FN)
            tpr = recall

        # 8. FPR：真实不属于当前类，却被错误预测成当前类的比例
        if FP + TN == 0:
            fpr = 0.0
        else:
            fpr = float(FP) / (FP + TN)

        # 9. F1：Precision 与 Recall 的调和平均
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        # 10. 将当前类别的所有指标保存到 class_metrics
        class_metrics[class_id] = {
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "TN": TN,
            "Precision": precision,
            "Recall": recall,
            "TPR": tpr,
            "FPR": fpr,
            "F1": f1,
        }

    return error_rate, accuracy, class_metrics


def plot_k_scores(k_scores):
    """绘制 K 值与 5 折平均准确率的关系曲线。"""
    k_values = list(k_scores.keys())
    scores = list(k_scores.values())

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, scores, marker="o")
    plt.xlabel("K")
    plt.ylabel("5-Fold Mean Accuracy")
    plt.title("K Selection by 5-Fold Cross Validation")
    plt.xticks(k_values)
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("k_selection.png", dpi=150)
    plt.show()


def plot_confusion_matrix(confusion):
    """绘制三分类混淆矩阵。"""
    class_names = ["Circle", "Square", "Triangle"]

    plt.figure(figsize=(6, 5))
    plt.imshow(confusion)

    plt.xticks(range(len(class_names)), class_names)
    plt.yticks(range(len(class_names)), class_names)

    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.title("Three-Class Confusion Matrix")

    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            plt.text(j, i, confusion[i][j], ha="center", va="center")

    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()


def main():
    trainX, trainY, testX, testY = loadDataSet()

    print("\n训练集大小：", trainX.shape)
    print("测试集大小：", testX.shape)

    k_list = [1, 3, 5, 7, 9]

    print("\n===== 5 折交叉验证选择 K =====")

    best_k, k_scores = cross_validation(
        trainX, trainY, k_list, n_folds=5
    )

    print("\n各 K 值平均准确率：")
    for k, score in k_scores.items():
        print("K=%d：%.2f%%" % (k, score * 100))

    print("\n最佳 K 值：", best_k)

    plot_k_scores(k_scores)

    print("\n===== 使用最佳 K 对测试集预测 =====")

    predictions = predict_test_set(
        testX, trainX, trainY, best_k
    )

    for i in range(len(testY)):
        print(
            "第%d张：真实=%s，预测=%s"
            % (
                i + 1,
                LABEL_NAMES[testY[i]],
                LABEL_NAMES[predictions[i]]
            )
        )

    confusion = build_confusion_matrix(
        testY, predictions, num_classes=3
    )

    print("\n===== 三分类混淆矩阵 =====")
    print("行：真实类别")
    print("列：预测类别")
    print(confusion)

    plot_confusion_matrix(confusion)

    error_rate, accuracy, class_metrics = calculate_metrics(confusion)

    print("\n===== 模型整体评价 =====")
    print("错误率：%.4f" % error_rate)
    print("准确率：%.4f" % accuracy)

    print("\n===== 各类别评价指标 =====")

    for class_id, metrics in class_metrics.items():
        print("\n类别：", LABEL_NAMES[class_id])

        print(
            "TP=%d  FP=%d  FN=%d  TN=%d"
            % (
                metrics["TP"],
                metrics["FP"],
                metrics["FN"],
                metrics["TN"]
            )
        )

        print("Precision：%.4f" % metrics["Precision"])
        print("Recall：%.4f" % metrics["Recall"])
        print("TPR：%.4f" % metrics["TPR"])
        print("FPR：%.4f" % metrics["FPR"])
        print("F1：%.4f" % metrics["F1"])


if __name__ == "__main__":
    main()
