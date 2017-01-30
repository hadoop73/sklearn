# coding:utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

"""
 用户id,时间戳,浏览行为数据,浏览子行为编号
"""
names = ['userid','time','browser_behavior','browser_behavior_number']
browse_history_train = pd.read_csv("../../pcredit/train/browse_history_train.txt",header=None)
browse_history_test = pd.read_csv("../../pcredit/test/browse_history_test.txt",header=None)

browse_history = pd.concat([browse_history_train,browse_history_test])
browse_history.columns = names

browse_history['browse_count'] = 1
#browse_history = browse_history.head(100)
users = list(browse_history.userid.unique())


# 按照时间统计
data = browse_history[['userid','time','browse_count']]

t = data.groupby(['userid','time']).agg(sum)
t.reset_index(inplace=True)

def time_m(u):
    d = {'userid':u}
    tu = t[t.userid==u]
    d['browse_max'] = tu['browse_count'].max()
    d['browse_min'] = tu['browse_count'].min()
    d['browse_mean'] = tu['browse_count'].mean()
    d['browse_median'] = tu['browse_count'].median()
    d['browse_var'] = tu['browse_count'].var()
    d['browse_std'] = tu['browse_count'].std()
    d['browse_count'] = tu['browse_count'].count()
    d['browse_max_min'] = d['browse_max'] - d['browse_min']
    print d
    return d

pool = Pool(12)
rst = pool.map(time_m,users)
pool.close()
pool.join()
Datas = pd.DataFrame(rst)
#print Data.head()
del rst,t,data


# 浏览数据统计
data = browse_history[['userid','browser_behavior','browse_count']]

t = data.groupby(['userid','browser_behavior']).agg(sum)
t.reset_index(inplace=True)

browser_behavior_tp = list(data.browser_behavior.unique())

# 统计 browser 类别数据
def browser_behavior_u(u):
    d = {"userid":u}
    ta = t.loc[t.userid == u, :]
    d['browser_data_max'] = ta['browse_count'].max()
    d['browser_data_min'] = ta['browse_count'].min()
    d['browser_data_mean'] = ta['browse_count'].mean()
    d['browser_data_median'] = ta['browse_count'].median()
    d['browser_data_var'] = ta['browse_count'].var()
    d['browser_data_std'] = ta['browse_count'].std()
    d['browser_data_count'] = ta['browse_count'].count()
    d['browser_data_max_min'] = d['browser_data_max'] - d['browser_data_min']
    #print ta
    for b in browser_behavior_tp:
        try:
            tb = ta.loc[ta.browser_behavior==b,'browse_count']
            d['browser_'+str(b)] = tb.iloc[0]
        except:
            d['browser_' + str(b)] = np.NAN
    print d
    return d

pool = Pool(12)

rst = pool.map(browser_behavior_u,users)
pool.close()
pool.join()
Data = pd.DataFrame(rst)
Datas = pd.merge(Datas,Data,on='userid')
del Data,rst,t,data




# 子行为统计
data = browse_history[['userid','browser_behavior_number','browse_count']]
t = data.groupby(['userid','browser_behavior_number']).agg(sum)
t.reset_index(inplace=True)


def browser_behavior_number_u(u):
    d = {"userid":u}
    ta = t.loc[t.userid == u, :]
    d['browser_behavior_max'] = ta['browse_count'].max()
    d['browser_behavior_min'] = ta['browse_count'].min()
    d['browser_behavior_mean'] = ta['browse_count'].mean()
    d['browser_behavior_median'] = ta['browse_count'].median()
    d['browser_behavior_var'] = ta['browse_count'].var()
    d['browser_behavior_std'] = ta['browse_count'].std()
    d['browser_behavior_count'] = ta['browse_count'].count()
    d['browser_behavior_max_min'] = d['browser_behavior_max'] - d['browser_behavior_min']
    for b in [1,4,5,6,7,8,10]:
        try:
            tb = ta.loc[t.browser_behavior_number==b,'browse_count']
            d['browser_behavior_number_'+str(b)] = tb.iloc[0]
        except:
            d['browser_behavior_number_' + str(b)] = np.NAN
    print d
    return d


pool = Pool(12)
rst = pool.map(browser_behavior_number_u,users)
pool.close()
pool.join()
Data = pd.DataFrame(rst)
Datas = pd.merge(Datas,Data,on='userid')
del Data,rst,data

print Datas.head()
print Datas.shape

Datas.to_csv('../data/train/browser_history_a.csv',index=None)

