# coding:utf-8



import numpy as np
import pandas as pd
from pandas import DataFrame,Series
import seaborn as sns
import matplotlib.pyplot as plt


bill_data = pd.read_csv('data/train/bill_diff_with0.csv')

bill_data.loc[:, 'repay_sub'] = bill_data.loc[:, 'pre_amount_of_bill'] - bill_data.loc[:, 'pre_repayment']
bill_data.loc[:, 'repay_sub_now'] = bill_data.loc[:, 'amount_of_bill'] - bill_data.loc[:, 'pre_repayment']

names_loan_time = ['userid','loan_time']
loan_time_train = pd.read_csv("../../pcredit/train/loan_time_train.txt",header=None)
loan_time_test = pd.read_csv("../../pcredit/test/loan_time_test.txt",header=None)
loan_time = pd.concat([loan_time_train,loan_time_test],axis=0)
loan_time.columns = names_loan_time
del loan_time_train,loan_time_test

bill_data = pd.merge(bill_data,loan_time,on='userid',how='left')
del loan_time,names_loan_time

users = list(bill_data.userid.unique())

cols = ['pre_amount_of_bill', 'pre_repayment', 'credit_amount',
        'amount_of_bill_left', 'least_repayment', 'consume_amount',
        'amount_of_bill', 'adjust_amount', 'circ_interest', 'avail_amount',
        'prepare_amount']
sts = ['_rate', '_min', '_max', '_median', '_mean', '_std', '_cnt', '_max_min']

# 统计每个用户，放款前后的信用卡使用情况
def test_0(user):
    """
    统计每个用户，放款前后的信用卡使用情况,删除了重复记录
    fileName = 'data/train/bill_diff0.csv'  这个是删除 0 的情况
    :param user:
    :return:
    """
    d = {'userid': user}
    bills = bill_data[bill_data.userid == user]
    # 统计每列总的情况
    # 只统计上期账单金额，上期还款金额，消费次数，信用额度,账单余额（账单余额越大越保险） 5 个列
    for col in ['pre_amount_of_bill','pre_repayment','consume_amount','credit_amount','amount_of_bill_left']:
        billdt = bills[bills[col]!=0]
        t = billdt[col]
        d[col + '_min'] = t.min()
        d[col + '_max'] = t.max()
        d[col + '_median'] = t.median()
        d[col + '_mean'] = t.mean()
        d[col + '_std'] = t.std()
        d[col + '_max_min'] = t.max() - t.min()

        # 统计每列放款前的情况
        t = billdt[billdt.time <= billdt.loan_time][col]
        d[col + '_min_lt'] = t.min()
        d[col + '_max_lt'] = t.max()
        d[col + '_median_lt'] = t.median()
        d[col + '_mean_lt'] = t.mean()
        d[col + '_std_lt'] = t.std()
        d[col + '_max_min_lt'] = t.max() - t.min()

        # 统计每列放款后的情况
        t = billdt[billdt.time > billdt.loan_time][col]
        d[col + '_min_gt'] = t.min()
        d[col + '_max_gt'] = t.max()
        d[col + '_median_gt'] = t.median()
        d[col + '_mean_gt'] = t.mean()
        d[col + '_std_gt'] = t.std()
        d[col + '_max_min_gt'] = t.max() - t.min()

        # 统计放款前后的差距
        d[col + '_min_gt_lt'] = d[col + '_min_gt'] - d[col + '_min_lt']
        d[col + '_max_gt_lt'] = d[col + '_max_gt'] - d[col + '_max_lt']
        d[col + '_mean_gt_lt'] = d[col + '_mean_gt'] - d[col + '_mean_lt']
        d[col + '_median_gt_lt'] = d[col + '_median_gt'] - d[col + '_median_lt']
        d[col + '_max_min_gt_lt'] = d[col + '_max_min_gt'] - d[col + '_max_min_lt']
    print d
    return d

def test_(user):
    """
    统计每个用户，放款前后的信用卡使用情况
    fileName = 'data/train/bill_data_time2.csv' 这个是没有删除 0 的情况

    :param user:
    :return:
    """
    d = {'userid':user}
    bills = bill_data[bill_data.userid==user]
    # 统计每列总的情况
    for col in cols:
        t = bills[col]
        d[col+'_min'] = t.min()
        d[col + '_max'] = t.max()
        d[col + '_median'] = t.median()
        d[col + '_mean'] = t.mean()
        d[col + '_std'] = t.std()
        d[col + '_max_min'] = t.max() - t.min()

        # 统计每列放款前的情况
        t = bills[bills.time<=bills.loan_time][col]
        d[col + '_min_lt'] = t.min()
        d[col + '_max_lt'] = t.max()
        d[col + '_median_lt'] = t.median()
        d[col + '_mean_lt'] = t.mean()
        d[col + '_std_lt'] = t.std()
        d[col + '_max_min_lt'] = t.max() - t.min()

        # 统计每列放款后的情况
        t = bills[bills.time > bills.loan_time][col]
        d[col + '_min_gt'] = t.min()
        d[col + '_max_gt'] = t.max()
        d[col + '_median_gt'] = t.median()
        d[col + '_mean_gt'] = t.mean()
        d[col + '_std_gt'] = t.std()
        d[col + '_max_min_gt'] = t.max() - t.min()

        # 统计放款前后的差距
        d[col + '_min_gt_lt'] = d[col + '_min_gt'] - d[col + '_min_lt']
        d[col + '_max_gt_lt'] = d[col + '_max_gt'] - d[col + '_max_lt']
        d[col + '_mean_gt_lt'] = d[col + '_mean_gt'] - d[col + '_mean_lt']
        d[col + '_median_gt_lt'] = d[col + '_median_gt'] - d[col + '_median_lt']
        d[col + '_max_min_gt_lt'] = d[col + '_max_min_gt'] - d[col + '_max_min_lt']
    print d
    return d


stage = ['stg'+str(i)+"_" for i in range(1,11)]
bill_data.loc[bill_data['consume_amount']>20,'consume_amount'] = 20


# 对每列数据进行分段
cols = ['time','repay_sub_now','repay_sub','pre_amount_of_bill','pre_repayment','consume_amount','credit_amount','amount_of_bill_left']


split_point_col = {}

for col in cols:
    t = bill_data[col].describe(percentiles=[0.1,0.2,0.3,0.4,0.5, 0.6,0.7,0.8,0.9])
    split_point_col[col] = [0,int(t['10%']), int(t['20%']),int(t['30%']), int(t['40%']),int(t['50%']),
                            int(t['60%']),int(t['70%']), int(t['80%']),int(t['90%']), 1e11]

def data_cut10(u):
    """
    所有的列数据被切分为 10 分，然后统计   fileName = 'data/train/data_cut10.csv'
    需要注意的是 NA 怎么处理
    :param u:
    :return:
    """
    d = {'userid':u}
    data = bill_data[bill_data.userid == u]
    data = data[cols]
    for col in cols:
        for i in range(10):
            stg = stage[i]
            di = data[(split_point_col[col][i] < (data.time)) & ((data.time) < split_point_col[col][i + 1])]
            d[stg + col + '_cnt'] = di[col].count()
            d[stg + col + '_min'] = di[col].min()
            d[stg + col + '_max'] = di[col].max()
            d[stg + col + '_mean'] = di[col].mean()
            if col not in ['time']:
                d[stg + col + '_std'] = di[col].std()
    #  应该根据用户的时间来切分
    print d
    return d


# 每列获取用户 3 个最小的数据，3个最大的数据
"""
https://www.rong360.com/gl/2014/03/17/35423.html
"""
def test_min_max3(u):
    """
    最大，最小的 3组数据  fileName = 'data/train/bill_diff_min_max3.csv'
    :param u:
    :return:
    """
    d = {'userid':u}
    data = bill_data[bill_data.userid == u]
    data = data[['repay_sub_now','repay_sub','pre_amount_of_bill','pre_repayment','consume_amount','credit_amount','amount_of_bill_left']]
    cols = data.columns
    for col in cols:
        d[col+'_min'] = data[col].min()
        d[col + '_max'] = data[col].max()
        d[col + '_mean'] = data[col].mean()
        d[col + '_median'] = data[col].median()
        d[col + '_std'] = data[col].std()
    #print data
    for col in cols:
        t = data.sort(col,ascending=True)
        t.reset_index(drop=True,inplace=True)
        for i in [0,1,2]:
            for c in cols:
                try:
                    d[col + "_" + c + "_" + str(i) + "_min"] = t.loc[i,c]
                except:
                    d[col + "_" + c + "_" + str(i) + "_min"] = np.NaN

    for col in cols:
        t = data.sort(col,ascending=False)
        t.reset_index(drop=True,inplace=True)
        for i in [0, 1, 2]:
            for c in cols:
                try:
                    d[col + "_" + c + "_" + str(i) + "_max"] = t.loc[i, c]
                except:
                    d[col + "_" + c + "_" + str(i) + "_max"] = np.NaN
    del data
    print d
    return d


# 分 5 个时间段 http://stackoverflow.com/questions/31593201/pandas-iloc-vs-ix-vs-loc-explanation
def test_time5(u):
    """
    按照自己的时间进行了分段   fileName = 'data/train/bill_diff_time5.csv'
    信用卡的最小时间，信用卡的张数
    # 统计信用额度，最小值，最大值，并分时间段统计，统计两个时间段的增加情况，按照数值分段统计
    消费次数记录，可以与本期账单相除，也就是每次的消费习惯，可以统计消费次数多于 1 记录
    统计上期还款金额，上期账单金额，也按照数据分段统计
    本期账单金额，上期账单金额，最少还款金额一样的，可以只统计一个 可以只统计一个
    :param u:
    :return:
    """
    # 获得所有的用户
    d = {'userid': u}
    data = bill_data[bill_data.userid == u]
    d['credit_time_min'] = data['time'].min()   # 信用卡创立最早时间
    d['credit_record_n'] = data['time'].count()  # 记录条数
    #  应该根据用户的时间来切分
    t = data['time'].describe(percentiles=[0.2, 0.4, 0.6,0.8,])
    split_point = [0,  int(t['20%']),  int(t['40%']), int(t['60%'])
        , int(t['80%']),1e11]
    # 统计每次消费多少钱
    data['amount_per_n'] = 1.0*data['amount_of_bill'] / (1+data['consume_amount'])
    # 还款记录情况,上期还款金额，上期账单
    data['repay_sub'] = data['pre_amount_of_bill'] - data['pre_repayment']
    for col in ['repay_sub','amount_per_n','pre_amount_of_bill','pre_repayment','consume_amount','credit_amount','amount_of_bill_left']:
         d[col + '_min_all'] = data[col].min()
         d[col + '_max_all'] = data[col].max()
         d[col + '_max_min_all'] = d[col + '_max_all'] - d[col + '_min_all']
         d[col + '_mean_all'] = data[col].mean()
         d[col+'stg_max_mean_all'] = -9999
         d[col + 'stg_max_min_all'] = -9999
         d[col + 'stg_max_max_all'] = -9999
         d[col + 'stg_max_max_min_all'] = -9999
         for i in range(5):
                stg = stage[i]
                di = data[(split_point[i]<(data.time))&((data.time)<split_point[i+1])]
                #  计算每一个用户的 最小，最大，中值，平均值，方差，数量，以及最大值和最小值的差
                d[stg+col+'_min_t5'] = di[col].min()
                d[stg + col + '_min_t5_all'] = d[stg + col + '_min_t5'] - d[col + '_min_all']
                d[col + 'stg_max_min_all'] = max(d[col + 'stg_max_min_all'],d[stg + col + '_min_t5_all'])

                d[stg+col+'_max_t5'] = di[col].max()
                d[stg + col + '_max_t5_all'] = d[col+'_max_all'] - d[stg + col + '_max_t5']
                d[col + 'stg_max_max_all'] = max(d[col + 'stg_max_max_all'], d[stg + col + '_max_t5_all'])

                d[stg+col+'_mean_t5'] = di[col].mean()
                d[stg + col + '_mean_t5_all'] = d[stg + col + '_mean_t5'] - d[col + '_mean_all']
                d[col + 'stg_max_mean_all'] = max(d[col + 'stg_max_mean_all'],d[stg + col + '_mean_t5_all'])

                d[stg+col+'_std_t5'] = di[col].std()
                d[stg+col+'_var_t5'] = di[col].var()
                d[stg+col+'_max_min_t5'] = di[col].max() - di[col].min()
                d[stg + col + '_max_min_t5_all'] = d[col+'_max_min_all'] - d[stg + col + '_max_min_t5']
                d[col + 'stg_max_max_min_all'] = max(d[stg+col+'_max_min_t5_all'],d[col + 'stg_max_max_min_all'])
                del di
    del data
    print d
    return d

def multi():
    from multiprocessing import Pool
    pool = Pool(8)
    u = users[:100]
    #  获得 split time 的数据统计信息
    rst = pool.map(data_cut10,users)
    pool.close()
    pool.join()
    features = pd.DataFrame(rst)
    fileName = 'data/train/data_cut10.csv'
    #  获得
    print features.head()
    print "\t", fileName
    print features.shape
    features.to_csv(fileName,index=None)

def merge(da="",db=""):
    a = pd.read_csv('data/train/bill_diff_time5.csv')
    b = pd.read_csv('data/train/bill_diff0.csv')
    d = pd.merge(a,b,on='userid',how='outer')
    print d.head()
    print d.shape
    d.to_csv("data/bill_merge.csv",index=None)

if __name__=="__main__":
    import warnings
    warnings.simplefilter('ignore')
    multi()
    #test_min_max3(6965)






