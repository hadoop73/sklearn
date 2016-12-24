# coding:utf-8



import numpy as np
import pandas as pd
from pandas import DataFrame,Series

import matplotlib.pyplot as plt


"""
 用户id,账单时间戳,银行id,上期账单金额,上期还款金额,信用卡额度,
 本期账单余额,本期账单最低还款额,消费笔数,本期账单金额,调整金额,
 循环利息,可用金额,预借现金额度,还款状态
"""

names = ["id","time","bank_id","pre_amount_of_bill","pre_repayment","credit_amount",\
         "amount_of_bill_left","least_repayment","consume_amount","amount_of_bill","adjust_amount",\
         "circ_interest","avail_amount","prepare_amount","repayment_state"]

bill_train = pd.read_csv("../../pcredit/train/bill_detail_train.txt",header=None)
bill_train.columns=names

columns = ['credit_amount',"consume_amount","avail_amount","circ_interest","prepare_amount"]
new_bill_train_mean = pd.pivot_table(bill_train,index=["id"],values=columns,aggfunc=np.mean)

ax = ['id','avail_amount','circ_interest','consume_amount','credit_amount','prepare_amount']

new_bill_train_mean['id'] = new_bill_train_mean.index
df = new_bill_train_mean[ax]
#df = DataFrame(df.values)
#df.columns = ax

df.to_csv('../data/train/bill_train.csv',index=None)


bill_test = pd.read_csv("../../pcredit/test/bill_detail_test.txt",header=None)

bill_test.columns=names
new_bill_test_mean = pd.pivot_table(bill_test,index=["id"],values=columns,aggfunc=np.mean)

new_bill_test_mean['id'] = new_bill_test_mean.index
df_test = new_bill_test_mean[ax]

df_test.to_csv('../data/test/bill_test.csv',index=None)









