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


names_loan_time = ['userid','loan_time']
loan_time_train = pd.read_csv("../../pcredit/train/loan_time_train.txt",header=None)
loan_time_test = pd.read_csv("../../pcredit/test/loan_time_test.txt",header=None)

loan_time = pd.concat([loan_time_train,loan_time_test],axis=0)

loan_time.columns = names_loan_time

loan_time = loan_time.set_index('userid')


import warnings
warnings.filterwarnings('ignore')  #  忽略警告

# 1） 统计收入支出频率，平均值，最大值，最小值，中值
# 2） 统计分段数据

#users = list(bank_detail.userid.unique())

def bank_data_get(data=bank_detail,sf='_eq'):

    bank_type_amount = data[['userid','extype','examount']]
    bank_type_amount['bank_type_amount_n'] = 1

    #  统计收入支出的次数
    bank_type_amount_nsum = bank_type_amount.groupby(['userid','extype'])['bank_type_amount_n'].sum()
    bank_type_amount_sum = bank_type_amount_nsum.unstack()
    bank_type_amount_sum.columns = ["extype##0{}".format(sf),"extype##1{}".format(sf)]

    #  统计收入支出的频率
    bank_type_amount_sum['extype_sum{}'.format(sf)] = bank_type_amount_sum['extype##0{}'.format(sf)] + bank_type_amount_sum['extype##1{}'.format(sf)]

    bank_type_amount_sum['extype##00{}'.format(sf)] = bank_type_amount_sum['extype##0{}'.format(sf)] / bank_type_amount_sum['extype_sum{}'.format(sf)]

    bank_type_amount_sum['extype##11{}'.format(sf)] = bank_type_amount_sum['extype##1{}'.format(sf)] / bank_type_amount_sum['extype_sum{}'.format(sf)]

    DATAS = bank_type_amount_sum


    #  输入支出的平均额度的最大最小值
    df = pd.pivot_table(data,index=['userid','extype'],values=['examount'],aggfunc=[np.max,np.min,np.median,np.var,np.std])
    dfun = df.unstack()
    sts = ['_min','_max','_median','_mean','_std']
    cols = ['examount_'+str(i)+s+sf for i in [0,1] for s in sts]
    dfun.columns = cols
    dfun['examount_0_max_min{}'.format(sf)] = dfun['examount_0_max{}'.format(sf)] - dfun['examount_0_min{}'.format(sf)]
    dfun['examount_1_max_min{}'.format(sf)] = dfun['examount_1_max{}'.format(sf)] - dfun['examount_1_min{}'.format(sf)]

    DATAS = DATAS.join(dfun)
    DATAS = DATAS.fillna(0)
    return DATAS

def get_data_lt(data=bank_detail):
    users = list(bank_detail.userid.unique())
    datas = pd.DataFrame(columns=data.columns)

    for u in users:
        a = bank_detail[bank_detail.userid == u]
        b = a[a.time < loan_time.iloc[u].loan_time]
        datas = pd.concat([datas,b],axis=0)
    return datas

def get_data_gt(data=bank_detail):
    users = list(bank_detail.userid.unique())
    datas = pd.DataFrame(columns=data.columns)

    for u in users:
        a = bank_detail[bank_detail.userid == u]
        b = a[a.time >= loan_time.iloc[u].loan_time]
        datas = pd.concat([datas,b],axis=0)
    return datas




if __name__=='__main__':

    users = list(bank_detail.userid.unique())
    DATA = pd.DataFrame(data=users,columns=['userid'])
    DATA = DATA.set_index('userid')
    #print DATA.head()

    rst = [(bank_detail,'_eq'),(get_data_lt(bank_detail),'_lt'),(get_data_gt(bank_detail),'_gt')]
    for d,sf in rst:
        DATA = DATA.join(bank_data_get(data=d,sf=sf),how='outer')

    DATA['extype_sum_lt_rate'] = DATA['extype_sum_lt'] / DATA['extype_sum_eq']
    DATA['extype_sum_gt_rate'] = DATA['extype_sum_gt'] / DATA['extype_sum_eq']

    DATA['extype##0_lt_rate'] = DATA['extype##0_lt'] / DATA['extype##0_eq']
    DATA['extype##0_gt_rate'] = DATA['extype##0_gt'] / DATA['extype##0_eq']

    print DATA.head()

    fileName = '../data/train/bank_detail_split_time.csv'

    DATA.to_csv(fileName)

    print "new banks datas: "
    print '\t',fileName

    print "datas size : ",DATA.shape





