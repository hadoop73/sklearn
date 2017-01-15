# coding:utf-8



import numpy as np
import pandas as pd

names = ["userid", "time", "bank_id", "pre_amount_of_bill", "pre_repayment", "credit_amount", \
         "amount_of_bill_left", "least_repayment", "consume_amount", "amount_of_bill", "adjust_amount", \
         "circ_interest", "avail_amount", "prepare_amount", "repayment_state"]

bill_train = pd.read_csv("../../pcredit/train/bill_detail_train.txt", header=None)
bill_test = pd.read_csv("../../pcredit/test/bill_detail_test.txt", header=None)

bill_data = pd.concat([bill_train, bill_test])
bill_data.columns = names

stage = ['stg1_','stg2_','stg3_','stg4_','stg5_','stg6_','stg7_','stg8_','stg9_','stg10_']

cols = ['pre_amount_of_bill', 'pre_repayment', 'credit_amount',
        'amount_of_bill_left', 'least_repayment', 'consume_amount',
        'amount_of_bill', 'circ_interest', 'avail_amount',
        'prepare_amount']
sts = ['_min', '_max', '_median', '_mean', '_std', '_cnt', '_max_min']


features = pd.DataFrame(columns=['userid'] + [ st+s+p for p in sts  for st in stage for s in cols])


def test_(col):
    #  首先去除缺失值，获得分段
    bill_col = bill_data[['userid',col]]

    data = bill_col[bill_col[col] != 0]
    #  获得 10% 20% 到 90% 的数据分割点
    t = data[col].describe(percentiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    split_point = [0, int(t['10%']), int(t['20%']), int(t['30%']), int(t['40%']), int(t['50%']), int(t['60%']),
                   int(t['70%']), int(t['80%']), int(t['90%']), 1e11]

    # 获得所有的用户
    users = list(data.userid.unique())
    ft = [st+col+s  for s in sts for st in stage]
    ftures = pd.DataFrame(columns=ft)

    for u in users:

        d = {'userid':u}
        bill_user = bill_data[bill_data.userid==u]

        for i in range(10):
                stg = stage[i]
                di = bill_user[(split_point[i]<bill_user[col])&(bill_user[col]<split_point[i+1])]
                #  计算每一个用户的 最小，最大，中值，平均值，方差，数量，以及最大值和最小值的差
                d[stg+col+'_min'] = di[col].min()
                d[stg+col+'_max'] = di[col].max()
                d[stg+col+'_median'] = di[col].median()
                d[stg+col+'_mean'] = di[col].mean()
                d[stg+col+'_std'] = di[col].std()
                d[stg+col+'_cnt'] = di[col].count()
                d[stg+col+'_max_min'] = di[col].max() - di[col].min()
        this_tv_features = pd.DataFrame(d,index=[0])
        ftures = pd.concat([ftures,this_tv_features],axis=0)
    return ftures

from multiprocessing import Pool

rst = []
pool = Pool(12)
#  按照特征咧进行处理
for col in cols:
    print col
    rst.append(pool.apply_async(test_,args=(col,)))
pool.close()
pool.join()

rst = [i.get() for i in rst]

all_features = pd.merge(rst,on='userid',how='outer')
all_features.to_csv('../data/train/bill_detail_stage.csv',index=None)

print all_features.head()




















