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


def test_all():

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
    return DATAS


stage = ['stg1_','stg2_','stg3_','stg4_','stg5_']

# 时间分为了 5 段，求收入支出的统计信息
def test_(u):
    #  时间分 5 段
    t = bank_detail['time'].describe(percentiles=[0.2,0.4,0.6,0.8])
    split_point = [0,int(t['20%']),int(t['40%']),int(t['60%']),int(t['80%']),1e11]
    # 获得所有的用户
    d = {'userid': u}
    for e in [0,1]:
        bank_user = bank_detail[bank_detail.userid == u]
        bank_user = bank_user[bank_user.extype==e]
        for col in ['examount']:
            data = bank_user[bank_user[col] != 0]
            for i in range(5):
                    stg = stage[i]
                    di = data[(split_point[i]<(data.time))&((data.time)<split_point[i+1])]
                    #  计算每一个用户的 最小，最大，中值，平均值，方差，数量，以及最大值和最小值的差
                    d[stg+col+str(e)+'_min'] = di[col].min()
                    d[stg+col+str(e)+'_max'] = di[col].max()
                    d[stg+col+str(e)+'_median'] = di[col].median()
                    d[stg+col+str(e)+'_mean'] = di[col].mean()
                    d[stg+col+str(e)+'_std'] = di[col].std()
                    d[stg+col+str(e)+'_cnt'] = di[col].count()
                    d[stg+col+str(e)+'_max_min'] = di[col].max() - di[col].min()
    #ftures = pd.DataFrame(d,index=[0])
    print "add userid: ", u
    return d

names_loan_time = ['userid','loan_time']
loan_time_train = pd.read_csv("../../pcredit/train/loan_time_train.txt",header=None)
loan_time_test = pd.read_csv("../../pcredit/test/loan_time_test.txt",header=None)

loan_time = pd.concat([loan_time_train,loan_time_test],axis=0)

loan_time.columns = names_loan_time

bank_detail = pd.merge(bank_detail,loan_time,on='userid')

# 统计放款前后的收入支出的统计信息
def test_split2(u):

    # 获得所有的用户
    d = {'userid': u}
    try:
        bank_users = bank_detail[bank_detail.userid == u]

        for e in [0,1]:

            bank_user = bank_users[bank_users.extype==e]

            data_gt = bank_user.loc[bank_user.time >= bank_user.loan_time,:]
            for col in ['examount']:
                di = data_gt.loc[data_gt[col] != 0,:]
                if di.shape[0] >= 2:
                    #  计算每一个用户的 最小，最大，中值，平均值，方差，数量，以及最大值和最小值的差
                    d[col+str(e)+'_min'+"_gt"] = di[col].min()
                    d[col+str(e)+'_max'+"_gt"] = di[col].max()
                    d[col+str(e)+'_median'+"_gt"] = di[col].median()
                    d[col+str(e)+'_mean'+"_gt"] = di[col].mean()
                    d[col+str(e)+'_std'+"_gt"] = di[col].std()
                    d[col+str(e)+'_cnt'+"_gt"] = di[col].count()
                    d[col+str(e)+'_max_min'+"_gt"] = di[col].max() - di[col].min()
                else:
                    d[col + str(e) + '_min' + "_gt"] = np.NaN
                    d[col + str(e) + '_max' + "_gt"] = np.NaN
                    d[col + str(e) + '_median' + "_gt"] = np.NaN
                    d[col + str(e) + '_mean' + "_gt"] = np.NaN
                    d[col + str(e) + '_std' + "_gt"] = np.NaN
                    d[col + str(e) + '_cnt' + "_gt"] = np.NaN
                    d[col + str(e) + '_max_min' + "_gt"] = np.NaN

            data_lt = bank_user.loc[bank_user.time < bank_user.loan_time,:]
            for col in ['examount']:
                di = data_lt.loc[data_lt[col] != 0, :]
                if di.shape[0] >= 2:
                    #  计算每一个用户的 最小，最大，中值，平均值，方差，数量，以及最大值和最小值的差
                    d[col + str(e) + '_min' + "_lt"] = di[col].min()
                    d[col + str(e) + '_max' + "_lt"] = di[col].max()
                    d[col + str(e) + '_median' + "_lt"] = di[col].median()
                    d[col + str(e) + '_mean' + "_lt"] = di[col].mean()
                    d[col + str(e) + '_std' + "_lt"] = di[col].std()
                    d[col + str(e) + '_cnt' + "_lt"] = di[col].count()
                    d[col + str(e) + '_max_min' + "_lt"] = di[col].max() - di[col].min()
                else:
                    d[col + str(e) + '_min' + "_lt"] = np.NaN
                    d[col + str(e) + '_max' + "_lt"] = np.NaN
                    d[col + str(e) + '_median' + "_lt"] = np.NaN
                    d[col + str(e) + '_mean' + "_lt"] = np.NaN
                    d[col + str(e) + '_std' + "_lt"] = np.NaN
                    d[col + str(e) + '_cnt' + "_lt"] = np.NaN
                    d[col + str(e) + '_max_min' + "_lt"] = np.NaN

        #ftures = pd.DataFrame(d,index=[0])
        print "add userid: ", u
        print d
    except Exception:
        print Exception
        print "userid: ",u
    return d


def test_split2x(u):
    try:
        return test_split2(u)
    except:
        pass



# 统计放款前后的时间差值
def time_split2(u):
    d = {'userid':u}
    bank_users = bank_detail[bank_detail.userid == u]
    for e in [0,1]:
        try:
            edata = bank_users[bank_users.extype==e]
            tdata = edata[edata.time!=0]
            dtime = tdata['time']-tdata['loan_time']
            d['bank_time_min_'+str(e)] = dtime.min()
            d['bank_time_max_'+str(e)] = dtime.max()
        except:
            d['bank_time_min_'+str(e)] = -9999
            d['bank_time_max_'+str(e)] = -9999
    print d
    return d



def new_data():
    from multiprocessing import Pool, Queue, Lock
    pool = Pool(5)
    #  按照特征咧进行处理

    users = list(bank_detail.userid.unique())

    rst = pool.map(time_split2, users)
    pool.close()
    pool.join()

    df = pd.DataFrame(rst)
    # df = df.set_index('userid')

    print df.head()

    # bank_detail
    # fileName = '../data/train/bank_detail_stage.csv'
    #  bank 数据按照 放款前后进行统计
    # fileName = '../data/train/bank_detail_split2.csv'
    # 统计时间的信息
    fileName = '../data/train/bank_time_2.csv'
    print df.head()

    print "new banks datas: "
    print '\t', fileName
    print "datas size : ", df.shape

    df.to_csv(fileName)

if __name__=='__main__':
    new_data()







