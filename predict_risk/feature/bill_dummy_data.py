# coding:utf-8


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


bill_train = pd.read_csv("../../pcredit/train/bill_detail_train.txt",header=None)
bill_test = pd.read_csv("../../pcredit/test/bill_detail_test.txt",header=None)
bill_data = pd.concat([bill_train,bill_test])

names = ["userid", "time", "bank_id", "pre_amount_of_bill", "pre_repayment", "credit_amount", \
         "amount_of_bill_left", "least_repayment", "consume_amount", "amount_of_bill", "adjust_amount", \
         "circ_interest", "avail_amount", "prepare_amount", "repayment_state"]
bill_data.columns = names

#  选择要处理的列
columns = ['userid','credit_amount',"consume_amount","avail_amount","circ_interest","prepare_amount"]
new_bill_data = bill_data[columns]
new_bill_data.head()


#  new_bill_data  的透视图
new_bill_data_mean = pd.pivot_table(new_bill_data,index=["userid"],values=columns,aggfunc=np.mean)
#  bill_detail  数据中 userid == 1 的数据没有
new_bill_data_mean.head()


#  对 new_bill_data_mean 的所有列数据进行标签编码
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
cols = new_bill_data_mean.columns
def transforLabel(datas,cols):
    for col in cols:
        data = datas[col]
        le.fit(data)
        datas[col] = le.transform(data)
    return datas
bill_label_data = transforLabel(new_bill_data_mean,cols)


s = pd.cut(bill_label_data['avail_amount'],20)
d = pd.get_dummies(s)
d.columns = ["{}#{}".format('avail_amount',i) for i in range(20)]


#  构建哑变量
#print bill_label_data.head()
cols = bill_label_data.columns
def dummyData(datas,cols):
    for col in cols:
        s = pd.cut(datas[col],20)
        d = pd.get_dummies(s)
        d.columns = ["{}#{}".format(col,i) for i in range(20)]
        datas.drop(col,axis = 1,inplace = True)
        datas = datas.join(d)
    return datas
bill_dummy_data = dummyData(bill_label_data.copy(),cols)


bill_dummy_data.to_csv("../data/train/bill_dummy_data.csv")


























