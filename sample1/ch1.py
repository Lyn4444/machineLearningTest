import numpy as np
import matplotlib.pyplot as plt

x, y = [], []

for i in open("prices.txt", "r"):
    _x, _y = i.split(",")
    x.append(float(_x))
    y.append(float(_y))

x, y = np.array(x), np.array(y)

x = (x - x.mean()) / x.std()

# 在（-2，4）区间取100个点作为画图基础
x0 = np.linspace(-2, 4, 100)


# n 是模型多项式次数
# 返回的模型根据输入的下（默认是x0）， 返回相对的预测的y
def getModel(n):
    return lambda input_x=x0: np.polyval(np.polyfit(x, y, n), input_x)


# 根据参数的n， 输入的x, y 返回相对于的损失
def getCost(n, input_x, input_y):
    return 0.5 * ((getModel(n)(input_x) - input_y) ** 2).sum()


test_set = (1, 4, 10)

# 损失看起来n=10最好，但是图看起来n=1最好， 可知n=10是 过拟合
for i in test_set:
    print(getCost(i, x, y))

plt.figure()
plt.scatter(x, y, c="g", s=20)
for test in test_set:
    plt.plot(x0, getModel(test)(), label="degree = {}".format(test))

plt.xlim(-2, 4)

plt.ylim(1e5, 8e5)
plt.legend()
plt.show()
