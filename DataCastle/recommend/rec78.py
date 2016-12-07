#coding:utf-8



import pandas as pd
import numpy as np

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

rate_rank = train.groupby('uid').mean().loc[:, ['score']].iloc[:, -1]
rate_rank = pd.DataFrame(np.int32((rate_rank * 2).values), index=rate_rank.index, columns=['group'])
rate_rank_des = rate_rank.reset_index()

train_plus = pd.merge(train, rate_rank_des, how='left', on='uid')
test_plus = pd.merge(test, rate_rank_des, how='left', on='uid')
res = train_plus.groupby(['iid', 'group']).mean().reset_index().loc[:, ['iid', 'group', 'score']]
result6 = pd.merge(test_plus, res, how='left', on=['iid', 'group']).filllna(3.0)