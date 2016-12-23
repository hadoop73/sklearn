# coding:utf-8



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 用户id,性别,职业,教育程度,婚姻状态,户口类型
names = ['id','sex','job','edu','marriage','account']
user_info_train = pd.read_csv("../../pcredit/train/user_info_train.txt",header=None)
user_info_train.columns=names

#  写入文件
user_info_train.to_csv("../data/user_info_train.csv",index=None)

user_info_test = pd.read_csv("../../pcredit/test/user_info_test.txt",header=None)
user_info_test.columns=names

#  测试数据写入文件

user_info_test.to_csv("../data/user_info_test.csv",index=None)











