# coding:utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
 用户id,时间戳,浏览行为数据,浏览子行为编号
"""
names = ['userid','time','browser_behavior','browser_behavior_number']
browse_history_train = pd.read_csv("../../pcredit/train/browse_history_train.txt",header=None)
browse_history_test = pd.read_csv("../../pcredit/test/browse_history_test.txt",header=None)

browse_history = pd.concat([browse_history_train,browse_history_test])
browse_history.columns = names

#  1) 统计 browser_behavior_number 每个类别的次数
browse_history['count'] = np.ones(len(browse_history['browser_behavior_number'])) # 添加数据进行统计
browser_behavior_count = pd.pivot_table(browse_history,index=['userid','browser_behavior_number'],values=['count'],aggfunc=np.sum)

browser_behavior_number_count = browser_behavior_count.unstack()
browser_behavior_number_count = browser_behavior_number_count.fillna(0)
browser_behavior_number_count.columns = [ "count#{}".format(i) for i in range(1,12)]


browser_behavior_number_count.to_csv('../data/train/browse_history.csv')






