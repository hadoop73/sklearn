# coding:utf-8


import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report

from pandas import DataFrame

def  KS(train,target,model):
    # 避免过拟合，采用交叉验证，验证集占训练集20%，固定随机种子（random_state)
    train_X, test_X, train_y, test_y = train_test_split(train,
                                                        target,
                                                        test_size=0.2,
                                                        random_state=0)
    pred = model.predict(test_X)
    test_y = DataFrame(test_y)
    pred = DataFrame(pred)
    y = pd.join([test_y,pred])

    pass
















