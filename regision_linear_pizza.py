#-*- coding:utf-8 -*-

import matplotlib.pyplot as plt

def runplt():
    plt.figure()
    plt.title('pizza and price')
    plt.xlabel('radis')
    plt.ylabel('price')
    plt.axis([0,25,0,25])
    plt.grid(True)
    return plt

plt = runplt()
X = [[6],[8],[10],[14],[18]]
Y = [[7],[9],[13],[17.5],[18]]

plt.plot(X,Y,'k.')
plt.show()

from sklearn.linear_model import LinearRegression

# 创建并拟合模型
model = LinearRegression()
model.fit(X,Y)
print '预测一张12英寸匹萨价格: %.2f' % model.predict([12])[0]

# 通过预测一个 list 来获得预测值 list ,并绘制直线
plt = runplt()
plt.plot(X, Y, 'k.')
X2 = [[0], [10], [14], [25]]
model = LinearRegression()
model.fit(X, Y)
y2 = model.predict(X2)
plt.plot(X, Y, 'k.')
plt.plot(X2, y2, 'g-')
plt.show()


