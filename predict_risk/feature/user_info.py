# coding:utf-8



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 用户id,性别,职业,教育程度,婚姻状态,户口类型
names = ['userid','sex','job','edu','marriage','account']
user_info_train = pd.read_csv("../../pcredit/train/user_info_train.txt",header=None)
user_info_test = pd.read_csv("../../pcredit/test/user_info_test.txt",header=None)


user_info = pd.concat([user_info_train,user_info_test])

user_info.columns = names

user_info = pd.pivot_table(user_info,index=["userid"],values=names)
user_info.head()  #  bill_detail  数据中 userid == 1 的数据没有

#  生成哑变量
def dummyTranform(datas,cols):
    for col in cols:
        datas[col].astype('category')
        d = pd.get_dummies(datas[col])
        d = d.add_prefix("{}#".format(col))
        datas = datas.join(d)
        datas.drop(col,axis = 1,inplace = True)
    return datas
user_info_dummy = dummyTranform(user_info.copy(),user_info.columns)
#  测试数据写入文件

user_info_dummy.to_csv("../data/train/user_info_dummy.csv")











