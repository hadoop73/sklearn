# coding:utf-8



"""
银行流水记录bank_detail.txt。
共5个字段，其中，第2个字段，时间戳为0表示时间未知；
第3个字段，交易类型有两个值，1表示支出、0表示收入；
第5个字段，工资收入标记为1时，表示工资收入。

用户id,时间戳,交易类型,交易金额,工资收入标记
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

names = ['userid','time','extype','examount','mark']
bank_detail_train = pd.read_csv("../../pcredit/train/bank_detail_train.txt",header=None)
bank_detail_test = pd.read_csv("../../pcredit/test/bank_detail_test.txt",header=None)

bank_detail = pd.concat([bank_detail_train,bank_detail_test])
bank_detail.columns = names

df = pd.pivot_table(bank_detail,index=['userid','extype'],values=['examount'],aggfunc=np.mean)

dfun = df.unstack()
dfun['sub'] = dfun['examount'][0]-dfun['examount'][1]

dfun.columns = ['examount#0','examount#1','sub']

#  删除空缺数据
dfun = dfun.dropna()

#  对 new_bill_data_mean 的所有列数据进行标签编码
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
cols = dfun.columns
def transforLabel(datas,cols):
    for col in cols:
        data = datas[col]
        le.fit(data)
        datas[col] = le.transform(data)
    return datas
bank_label_data = transforLabel(dfun,cols)

#  构建哑变量
#print bill_label_data.head()
cols = bank_label_data.columns
def dummyData(datas,cols):
    for col in cols:
        s = pd.cut(datas[col],20)
        d = pd.get_dummies(s)
        d.columns = ["{}#{}".format(col,i) for i in range(20)]
        datas.drop(col,axis = 1,inplace = True)
        datas = datas.join(d)
    return datas
bank_dummy_data = dummyData(bank_label_data.copy(),cols)

bank_dummy_data.to_csv('../data/train/bank_dummy_data.csv')









