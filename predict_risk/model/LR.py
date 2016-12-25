# coding:utf-8

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


loan_data = pd.read_csv("../data/train_data.csv")

# overdue_train，这是我们模型所要拟合的目标
target = pd.read_csv('../../pcredit/train/overdue_train.txt',
                         header = None)
target.columns = ['userid', 'label']
target.index = target['userid']
target.drop('userid',
            axis = 1,
            inplace = True)
# 构建模型
# 分开训练集、测试集
train = loan_data.iloc[0: 55597, :]
test = loan_data.iloc[55597:, :]
# 避免过拟合，采用交叉验证，验证集占训练集20%，固定随机种子（random_state)
train_X, test_X, train_y, test_y = train_test_split(train,
                                                    target,
                                                    test_size = 0.2,
                                                    random_state = 0)

train_y = train_y['label']
test_y = test_y['label']
# 这里用Logistic回归
lr_model = LogisticRegression(C = 1.0,
                              penalty = 'l2')
lr_model.fit(train_X, train_y)
# 给出交叉验证集的预测结果，评估准确率、召回率、F1值
pred_test = lr_model.predict(test_X)
print classification_report(test_y, pred_test)
# 输出测试集用户逾期还款概率，predict_proba会输出两个概率，取‘1’的概率
pred = lr_model.predict_proba(test)
result = pd.DataFrame(pred)
result.index = test.index
result.columns = ['0', 'probability']
result.drop('0',
            axis = 1,
            inplace = True)
print result.head(5)
# 输出结果
result.to_csv('../data/result.csv')


















