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

"""
userid	time	extype	examount	mark
6965	5894316387	0	13.756664	0
6965	5894321388	1	13.756664	0
6965	5897553564	0	14.449810	0
6965	5897563463	1	10.527763	0
6965	5897564598	1	13.651303	0
"""

names = ['userid','time','extype','examount','mark']
bank_detail_train = pd.read_csv("../../pcredit/train/bank_detail_train.txt",header=None)
bank_detail_test = pd.read_csv("../../pcredit/test/bank_detail_test.txt",header=None)

bank_detail = pd.concat([bank_detail_train,bank_detail_test])
bank_detail.columns = names


import warnings
warnings.filterwarnings('ignore')  #  忽略警告

# 1） 统计收入支出频率，平均值，最大值，最小值，中值
# 2） 统计分段数据

#users = list(bank_detail.userid.unique())


bank_type_amount = bank_detail[['userid','extype','examount']]
bank_type_amount['bank_type_amount_n'] = 1


#  统计收入支出的次数
bank_type_amount_nsum = bank_type_amount.groupby(['userid','extype'])['bank_type_amount_n'].sum()
bank_type_amount_sum = bank_type_amount_nsum.unstack()
bank_type_amount_sum.columns = ["extype##0","extype##1"]

#  统计收入支出的频率
bank_type_amount_sum['extype_sum'] = bank_type_amount_sum['extype##0'] + bank_type_amount_sum['extype##1']

bank_type_amount_sum['extype#00'] = bank_type_amount_sum['extype##0'] / bank_type_amount_sum['extype_sum']

bank_type_amount_sum['extype##11'] = bank_type_amount_sum['extype##1'] / bank_type_amount_sum['extype_sum']

DATAS = bank_type_amount_sum


#  输入支出的平均额度的最大最小值
df = pd.pivot_table(bank_detail,index=['userid','extype'],values=['examount'],aggfunc=[np.max,np.min,np.median,np.var,np.std])
dfun = df.unstack()
sts = ['_min','_max','_median','_mean','_std']
cols = ['examount_'+str(i)+s for i in [0,1] for s in sts]
dfun.columns = cols
dfun['examount_0_max_min'] = dfun['examount_0_max'] - dfun['examount_0_min']
dfun['examount_1_max_min'] = dfun['examount_1_max'] - dfun['examount_1_min']


DATAS = DATAS.join(dfun)
DATAS = DATAS.fillna(0)


DATAS.to_csv('../data/train/bank_detail.csv')


print "new banks datas: "
print '\t','../data/train/bank_detail.csv'


print "datas size : ",DATAS.shape





