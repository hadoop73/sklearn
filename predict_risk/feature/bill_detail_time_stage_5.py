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

bill_data.loc[bill_data['consume_amount']>60,'consume_amount'] = 60
bill_data.loc[bill_data['prepare_amount']>25,'prepare_amount'] = 23

stage = ['stg1_','stg2_','stg3_','stg4_','stg5_']

cols = ['time','pre_amount_of_bill', 'pre_repayment', 'credit_amount',
        'amount_of_bill_left', 'least_repayment', 'consume_amount',
        'amount_of_bill', 'adjust_amount', 'circ_interest', 'avail_amount',
        'prepare_amount']
sts = ['_min', '_max', '_median', '_mean', '_std', '_cnt', '_max_min']

features = pd.DataFrame(columns=[s+col+st for st in sts for s in stage for col in cols])

from multiprocessing import Pool,Queue,Lock


def test_(u):

    #  获得 10% 20% 到 90% 的数据分割点
    t = bill_data['time'].describe(percentiles=[0.2,0.4,0.6,0.8])

    split_point = [0,int(t['20%']),int(t['40%']),int(t['60%']),int(t['80%']),1e11]

    # 获得所有的用户
    d = {'userid': u}

    for col in cols:
        bill_user = bill_data[bill_data.userid==u]
        data = bill_user[bill_user[col] != 0]

        for i in range(5):
                stg = stage[i]
                di = data[(split_point[i]<(data.time))&((data.time)<split_point[i+1])]
                if col in ['time']:
                    di[col] = di[col] % 2017
                #  计算每一个用户的 最小，最大，中值，平均值，方差，数量，以及最大值和最小值的差
                d[stg+col+'_min_t5'] = di[col].min()
                d[stg+col+'_max_t5'] = di[col].max()
                d[stg+col+'_median_t5'] = di[col].median()
                d[stg+col+'_mean_t5'] = di[col].mean()
                d[stg+col+'_std_t5'] = di[col].std()
                d[stg+col+'_cnt_t5'] = di[col].count()
                d[stg+col+'_var_t5'] = di[col].var()
                d[stg+col+'_max_min_t5'] = di[col].max() - di[col].min()
    #ftures = pd.DataFrame(d,index=[0])
    print "add userid: ", u
    print d
    return d



rst = []
pool = Pool(12)
#  按照特征咧进行处理

users = list(bill_data.userid.unique())

for u in users:
    #print "add userid: ",u
    rst.append(pool.apply_async(test_,args=(u,)))
pool.close()
pool.join()

rst = [i.get() for i in rst]

#features = pd.DataFrame(columns=[ st+s+p for p in sts  for st in stage for s in cols])

features = pd.DataFrame(rst)

#for i in range(l):
    #features.loc[i] =rst[i]
#    features.append(rst[i],ignore_index=True)

#features = reduce(lambda x,y:pd.concat([x,y],axis=0),rst)

#for i in rst[1:]:
#    features = pd.merge(features,i,on='userid',how='outer')
# 效果不错 0.44
#features.to_csv('../data/train/bill_detail_time_stage5.csv',index=None)


print features.head()
print features.shape


features.to_csv('../data/train/bill_detail_time_stage52.csv',index=None)

















