import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


class MyPerception:
    def __init__(self, alpha=0.2, times=1500):
        self.alpha = alpha
        self.times = times

    def sign(self, x):
        predict = []
        for i in range(x.shape[0]):
            if x[i]+self.bias > 0:
                predict.append(1)
            else:
                predict.append(-1)
        return predict

    def fit(self, x_train, y_train):    # 拟合
        y_train[y_train == 0] = -1
        self.weights = np.zeros(x_train.shape[1])      # 给w赋初值
        self.bias = 0.1     # 给b赋初值
        self.weights.reshape(-1, 1)    # 转为列向量
        for _ in range(self.times):
            y_predict = np.array(self.sign(np.dot(x_train, self.weights)))
            y_error = y_train[y_train != y_predict]     # 预测与真实中不同的，即为误分类点
            x_error = x_train[y_predict != y_train]
            if y_error.shape[0] != 0:       # 有误分类点，则进行更新
                number = np.arange(y_error.shape[0])
                index = np.random.choice(number)
                self.weights += self.alpha*y_error[index]*x_error[index]
                self.bias += self.alpha*y_error[index]

    def predict(self, x_train):     # 预测
        test_predict = self.sign(np.dot(x_train, self.weights))
        return test_predict


def picture(data, percept):   # 作图，查看训练集的数据分布
    plt.figure()
    plt.style.use('seaborn')
    fig, ax = plt.subplots()
    ax.set_title("picture of data")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for i in range(0, data.shape[0]):
        if data[i][0]+data[i][1] > 1:
            ax.scatter(data[i][0], data[i][1], color='red', s=20)    # 设置点的颜色大小
        else:
            ax.scatter(data[i][0], data[i][1], color='blue', s=20)
    x = np.linspace(0, 1, 50)
    y = -(percept.weights[0] * x + percept.bias) / percept.weights[1]
    plt.plot(x, y)
    plt.show()


def main():
    tensor = (np.random.rand(1000, 2))

    # 平行列表
    rearSum = np.sum(tensor, 1)  # 压缩列，对行求和
    parallelList = []
    for num in range(0, 1000):
        if rearSum[num] > 1:  # 和大于1，则在列表中添加1，否则为0
            parallelList.append(1)
        else:
            parallelList.append(0)
    list1 = []
    list2 = []


    Train = np.zeros((700, 2))
    Test = np.zeros((300, 2))
    tempList = random.sample(range(0, 1000), 1000)  # 在区间[0,1000)间的整数随机排列，放在tempList列表
    count = 0
    for tempNum in tempList[0:700]:
        Train[count][0] = tensor[tempNum][0]
        Train[count][1] = tensor[tempNum][1]
        list1.append(parallelList[tempNum])
        count += 1
    count = 0
    for tempNum in tempList[700:1000]:
        Test[count][0] = tensor[tempNum][0]
        Test[count][1] = tensor[tempNum][1]
        list2.append(parallelList[tempNum])
        count += 1

    trainLabel = np.array(list1)
    testLabel = np.array(list2)
    percept = MyPerception()
    percept.fit(Train, trainLabel)

    print("weights=", percept.weights, '\n')
    print("bias=", percept.bias, '\n')

    # 开始预测
    testPredict = np.array(percept.predict(Test))
    testPredict[testPredict == -1] = 0
    picture(Test, percept)

    print(classification_report(testLabel, testPredict))


if __name__ == '__main__':
    main()
