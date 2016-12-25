# coding:utf-8

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


loan_data = pd.read_csv("../data/train_data.csv")

loan_data.index = loan_data.userid
loan_data.drop(['userid'],axis=1,inplace=True)

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
train = loan_data.iloc[0: 55596, :]
test = loan_data.iloc[55596:, :]


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor


clf = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                          n_estimators=300)

clf = clf.fit(train, target)
result = clf.predict(test)

# 输出测试集用户逾期还款概率，predict_proba会输出两个概率，取‘1’的概率

result = pd.DataFrame(result)

print result.head()

result.index = test.index
result.columns = ['probability']

print result.head(5)
# 输出结果
result.to_csv('../data/result_Ad.csv')








