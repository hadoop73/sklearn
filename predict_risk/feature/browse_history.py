# coding:utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
 用户id,时间戳,浏览行为数据,浏览子行为编号
"""

def get_browser_datas():

    names = ['userid','time','browser_behavior','browser_behavior_number']
    browse_history_train = pd.read_csv("../../pcredit/train/browse_history_train.txt",header=None)
    browse_history_test = pd.read_csv("../../pcredit/test/browse_history_test.txt",header=None)

    browse_history = pd.concat([browse_history_train,browse_history_test])
    browse_history.columns = names

    browse_history['browse_count'] = 1

    browse_time = browse_history[['userid','time','browse_count']]
    browse_time_n = pd.pivot_table(browse_time,index=['userid','time'],values=['browse_count'],aggfunc=sum)

    browse_time_max = browse_time_n.max(level='userid')
    browse_time_max.columns = ['browse_time_max']

    browse_time_min = browse_time_n.min(level='userid')
    browse_time_min.columns = ['browse_time_min']

    browse_time_median = browse_time_n.median(level='userid')
    browse_time_median.columns = ['browse_time_median']

    browse_time_mean = browse_time_n.mean(level='userid')
    browse_time_mean.columns = ['browse_time_mean']

    browse_time_var = browse_time_n.var(level='userid')
    browse_time_var.columns = ['browse_time_var']


    browse_time_std = browse_time_n.std(level='userid')
    browse_time_std.columns = ['browse_time_std']


    browse_time_n_1 = browse_time_n.copy()
    browse_time_n_1['browser_time_n'] = 1
    browse_time_n_1.drop(['browse_count'],axis=1,inplace=True)
    #browse_time_n_1.head()

    browse_time_n_1_sum = browse_time_n_1.sum(level='userid')
    browse_time_n_1_sum.columns = ['browse_time_n_1_sum']

    browser_behavior = browse_history[['userid','browser_behavior','browse_count']]
    browser_behavior_g = pd.pivot_table(browser_behavior,index=['userid','browser_behavior'],values=['browse_count'],aggfunc=sum)

    browser_behavior_max = browser_behavior_g.max(level='userid')
    browser_behavior_max.columns = ['browser_behavior_max']

    browser_behavior_min = browser_behavior_g.min(level='userid')
    browser_behavior_min.columns = ['browser_behavior_min']

    #  1) 统计 browser_behavior_number 每个类别的次数
    browse_history['count'] = np.ones(len(browse_history['browser_behavior_number'])) # 添加数据进行统计
    browser_behavior_count = pd.pivot_table(browse_history,index=['userid','browser_behavior_number'],values=['count'],aggfunc=np.sum)

    browser_behavior_count_mean = browser_behavior_count.mean(level='userid')
    browser_behavior_count_mean.columns = ['browser_behavior_count_mean']

    browser_behavior_mean = browser_behavior_g.mean(level='userid')
    browser_behavior_mean.columns = ['browser_behavior_mean']

    browser_data = browser_behavior_mean.join([browse_time_max,  # 统计相同时间的平均次数
                                                       browse_time_min,
                                                       browse_time_mean,
                                                       browse_time_median,
                                                       browse_time_var,
                                                       browse_time_std,
                                                       browse_time_n_1_sum,  # 不同时间的记录条数统计
                                                       browser_behavior_max,
                                                       browser_behavior_min,
                                                       browser_behavior_count_mean
                                                       ])


    print "browser_data"
    print '\t','../data/train/browse_history.csv'


    print browser_data.shape

    browser_data.to_csv('../data/train/browse_history.csv')


names = ['userid', 'time', 'browser_behavior', 'browser_behavior_number']
browse_history_train = pd.read_csv("../../pcredit/train/browse_history_train.txt", header=None)
browse_history_test = pd.read_csv("../../pcredit/test/browse_history_test.txt", header=None)

browse_history = pd.concat([browse_history_train, browse_history_test])
browse_history.columns = names
browse_history['browse_count'] = 1
del browse_history_train, browse_history_test

stage = ['stg1_','stg2_','stg3_','stg4_','stg5_']

#  获得 10% 20% 到 90% 的数据分割点
t = browse_history['time'].describe(percentiles=[0.2, 0.4, 0.6, 0.8])

split_point = [0, int(t['20%']), int(t['40%']), int(t['60%']), int(t['80%']), 1e11]

# 统计总的浏览记录信息，以及分段时间的统计信息,还有时间的次数统计也就是 cnt，可以统计一下
def test_(u):
    # 获得所有的用户
    d = {'userid': u}
    data = browse_history[browse_history.userid == u]
    # 不分阶段,统计总的一个情况,组要还是一个时间上的统计
    brdata = data[['userid', 'time', 'browse_count']].groupby(['userid', 'time']).agg(sum)
    brdata.reset_index(inplace=True)
    d['browse_count' + '_min'] = brdata['browse_count'].min()
    d['browse_count' + '_max'] = brdata['browse_count'].max()
    d['browse_count' + '_mean'] = brdata['browse_count'].mean()
    d['browse_count' + '_median'] = brdata['browse_count'].median()
    d['browse_count' + '_std'] = brdata['browse_count'].std()
    d['browse_count' + '_count'] = brdata['browse_count'].count()
    d['browse_count' + '_var'] = brdata['browse_count'].var()
    d['log_cnt'] = brdata['browse_count'].sum()

    d['browse_count' + '_min'] = brdata['browse_count'].min()
    d['browse_count' + '_max'] = brdata['browse_count'].max()
    d['browse_count' + '_mean'] = brdata['browse_count'].mean()
    d['browse_count' + '_median'] = brdata['browse_count'].median()
    d['browse_count' + '_std'] = brdata['browse_count'].std()
    d['browse_count' + '_count'] = brdata['browse_count'].count()
    d['browse_count' + '_var'] = brdata['browse_count'].var()
    d['log_cnt'] = brdata['browse_count'].sum()
    del brdata

    for i in range(5):
                stg = stage[i]
                di = data[(split_point[i]<(data.time))&((data.time)<split_point[i+1])]
                #  计算每一个用户的 最小，最大，中值，平均值，方差，数量，以及最大值和最小值的差
                brdata = di[['userid', 'time', 'browse_count']].groupby(['userid', 'time']).agg(sum)
                brdata.reset_index(inplace=True)
                d[stg+'browse_count'+'_min'] = brdata['browse_count'].min()
                d[stg + 'browse_count' + '_max'] = brdata['browse_count'].max()
                d[stg + 'browse_count' + '_mean'] = brdata['browse_count'].mean()
                d[stg + 'browse_count' + '_median'] = brdata['browse_count'].median()
                d[stg + 'browse_count' + '_std'] = brdata['browse_count'].std()
                d[stg + 'browse_count' + '_count'] = brdata['browse_count'].count()
                d[stg + 'browse_count' + '_var'] = brdata['browse_count'].var()
                d[stg + 'log_cnt'] = brdata['browse_count'].sum()
                del brdata,di

    #ftures = pd.DataFrame(d,index=[0])
    print d
    return d


names_loan_time = ['userid','loan_time']
loan_time_train = pd.read_csv("../../pcredit/train/loan_time_train.txt",header=None)
loan_time_test = pd.read_csv("../../pcredit/test/loan_time_test.txt",header=None)
loan_time = pd.concat([loan_time_train,loan_time_test],axis=0)
loan_time.columns = names_loan_time

browse_history = pd.merge(browse_history,loan_time,on='userid')
browse_history['browse_count'] = 1

del loan_time

#  时间分为放款前后，进行统计，可以统计一下与放款时间的差值
def test_split_time(u):
    # 获得所有的用户
    d = {'userid': u}
    du = browse_history[browse_history.userid == u]
    data = du[du.time <= du.loan_time]
    # 不分阶段,统计总的一个情况,组要还是一个时间上的统计
    d['browse_count' + '_time'+'_lt'] = np.min(data['time']-data['loan_time'])
    brdata = data[['userid', 'time', 'browse_count']].groupby(['userid', 'time']).agg(sum)
    brdata.reset_index(inplace=True)

    d['browse_count' + '_min'+'_lt'] = brdata['browse_count'].min()
    d['browse_count' + '_max'+'_lt'] = brdata['browse_count'].max()
    d['browse_count' + '_mean'+'_lt'] = brdata['browse_count'].mean()
    d['browse_count' + '_median'+'_lt'] = brdata['browse_count'].median()
    d['browse_count' + '_std'+'_lt'] = brdata['browse_count'].std()
    d['browse_count' + '_count'+'_lt'] = brdata['browse_count'].count()
    d['browse_count' + '_var'+'_lt'] = brdata['browse_count'].var()
    d['log_cnt'+'_lt'] = brdata['browse_count'].sum()
    d['browse_count_max_min_lt'] = brdata['browse_count'].max() - brdata['browse_count'].min()
    del brdata,data

    data = du[du.time > du.loan_time]
    d['browse_count' + '_time' + '_gt'] = np.max(data['time'] - data['loan_time'])
    # 不分阶段,统计总的一个情况,组要还是一个时间上的统计
    brdata = data[['userid', 'time', 'browse_count']].groupby(['userid', 'time']).agg(sum)
    brdata.reset_index(inplace=True)
    d['browse_count' + '_min'+'_gt'] = brdata['browse_count'].min()
    d['browse_count' + '_max'+'_gt'] = brdata['browse_count'].max()
    d['browse_count' + '_mean'+'_gt'] = brdata['browse_count'].mean()
    d['browse_count' + '_median'+'_gt'] = brdata['browse_count'].median()
    d['browse_count' + '_std'+'_gt'] = brdata['browse_count'].std()
    d['browse_count' + '_count'+'_gt'] = brdata['browse_count'].count()
    d['browse_count' + '_var'+'_gt'] = brdata['browse_count'].var()
    d['log_cnt'+'_gt'] = brdata['browse_count'].sum()
    d['browse_count_max_min_gt'] = brdata['browse_count'].max() - brdata['browse_count'].min()

    del brdata, data,du
    print d
    return d

#  统计时间信息，时间的最大值，最小值，
def time_browse(u):
    pass


def multi_browser():

    from multiprocessing import Pool,Queue,Lock
    pool = Pool(4)

    users = list(browse_history.userid.unique())
    rst = pool.map(test_split_time,users)

    features = pd.DataFrame(rst)
    features.fillna(-9999,inplace=True)
    print features.head()
    print features.shape

    #features.to_csv('../data/train/browse_stage5.csv',index=None)  # 没有数量筛选
    #features.to_csv('../data/train/browse_stage5_gt2.csv',index=None)  # 5 分段数据筛选数据大于 2 的数据进行统计
    #features.to_csv('../data/train/browse_split_time_gt2.csv',index=None)  # 筛选数据大于 2 的数据进行统计
    features.to_csv('../data/train/browse_split_time.csv',index=None)

def merge_browser():
    d = pd.read_csv('../data/train/browse_stage5_gt2.csv')  # 5 分段数据筛选数据大于 2 的数据进行统计
    d1 = pd.read_csv('../data/train/browse_split_time_gt2.csv')   # 时间分为放款前后，筛选数据大于 2 的数据进行统计
    data = pd.merge(d,d1,on='userid')
    print data.head()
    print data.shape
    data.to_csv('../data/train/browser_all.csv',index=None)


if __name__=='__main__':
    multi_browser()








#browser_behavior_number_count.to_csv('../data/train/browse_history.csv')






