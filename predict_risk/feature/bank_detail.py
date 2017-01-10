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

dfun.to_csv('../data/train/bank_detail.csv')









