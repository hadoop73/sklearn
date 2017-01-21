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

stage = ['stg1_','stg2_','stg3_','stg4_','stg5_']

def test_(u):

    #  获得 10% 20% 到 90% 的数据分割点
    t = browse_history['time'].describe(percentiles=[0.2,0.4,0.6,0.8])

    split_point = [0,int(t['20%']),int(t['40%']),int(t['60%']),int(t['80%']),1e11]

    # 获得所有的用户
    d = {'userid': u}

    data = browse_history[browse_history.userid == u]

    for i in range(5):
                stg = stage[i]
                di = data[(split_point[i]<(data.time))&((data.time)<split_point[i+1])]
                #  计算每一个用户的 最小，最大，中值，平均值，方差，数量，以及最大值和最小值的差
                d[stg+'_cnt'] = di['time'].count()
                d[stg + 'log_cnt'] = np.log(1+di['time'].count())
    #ftures = pd.DataFrame(d,index=[0])
    print "add userid: ", u
    return d

rst = []
from multiprocessing import Pool,Queue,Lock
pool = Pool(4)

users = list(browse_history.userid.unique())
i=0
for u in users:

    rst.append(pool.apply_async(test_, args=(u,)))


rst = [i.get() for i in rst]

features = pd.DataFrame(rst)

features.to_csv('../data/train/browse_history_stage.csv',index=None)
print features.head()

#if __name__=='__main__':

    # 由上面函数生成的数据
    #browser_datas = pd.read_csv('../data/train/browse_history.csv')








#browser_behavior_number_count.to_csv('../data/train/browse_history.csv')






