# coding:utf-8

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def getDatas(dir='train_data_'):
    loan_data = pd.read_csv("../data/{}.csv".format(dir))

    loan_data.index = loan_data['userid']
    loan_data.drop('userid',axis=1,inplace=True)

    # overdue_train，这是我们模型所要拟合的目标
    target = pd.read_csv('../../pcredit/train/overdue_train.txt',
                         header=None)
    target.columns = ['userid', 'label']
    target.index = target['userid']
    target.drop('userid',axis=1,inplace=True)
    # 构建模型
    # 分开训练集、测试集
    train = loan_data.iloc[0: 55596, :]
    test = loan_data.iloc[55596:, :]
    t = test

    t['probability'] = 0.1
    t = t['probability']
    t.columns = ['userid','probability' ]
    t.to_csv('../data/res.csv')
    return train,target,test

if __name__=='__main__':
    getDatas()

def getXGBoostDatas(dir='train_data'):
    loan_data = pd.read_csv("../data/{}.csv".format(dir))

    #loan_data.index = loan_data['userid']
    #loan_data.drop('userid',axis=1,inplace=True)

    # overdue_train，这是我们模型所要拟合的目标
    target = pd.read_csv('../../pcredit/train/overdue_train.txt',
                         header=None)
    target.columns = ['userid', 'label']
    #target.index = target['userid']
    #target.drop('userid',axis=1,inplace=True)
    # 构建模型
    # 分开训练集、测试集
    train = loan_data.iloc[0: 55596, :]
    test = loan_data.iloc[55596:, :]
    return train,target,test



