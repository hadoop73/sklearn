# coding:utf-8



import numpy as np
import pandas as pd

names = ["userid", "time", "bank_id", "pre_amount_of_bill", "pre_repayment", "credit_amount", \
         "amount_of_bill_left", "least_repayment", "consume_amount", "amount_of_bill", "adjust_amount", \
         "circ_interest", "avail_amount", "prepare_amount", "repayment_state"]

bill_train = pd.read_csv("../../pcredit/train/bill_detail_train.txt", header=None)
bill_test = pd.read_csv("../../pcredit/test/bill_detail_test.txt", header=None)
bill_data = pd.concat([bill_train, bill_test])

del bill_train, bill_test
bill_data.columns = names


bill_data = bill_data[bill_data['time']>5840000000]
bill_data.loc[bill_data['consume_amount']>12,'consume_amount'] = 12
#bill_data.loc[bill_data['prepare_amount']>25,'prepare_amount'] = 23

names_loan_time = ['userid','loan_time']
loan_time_train = pd.read_csv("../../pcredit/train/loan_time_train.txt",header=None)
loan_time_test = pd.read_csv("../../pcredit/test/loan_time_test.txt",header=None)

loan_time = pd.concat([loan_time_train,loan_time_test],axis=0)
del loan_time_train,loan_time_test
loan_time.columns = names_loan_time

bill_data = pd.merge(bill_data,loan_time,on='userid')
del loan_time

stage = ['stg1_','stg2_','stg3_','stg4_','stg5_']

cols = ['time','pre_amount_of_bill', 'pre_repayment', 'credit_amount',
        'amount_of_bill_left', 'least_repayment', 'consume_amount',
        'amount_of_bill', 'adjust_amount', 'circ_interest', 'avail_amount',
        'prepare_amount']

from multiprocessing import Pool,Queue,Lock

#  获得 10% 20% 到 90% 的数据分割点
t = bill_data['time'].describe(percentiles=[0.2, 0.4, 0.6, 0.8])
split_point = [0, int(t['20%']), int(t['40%']), int(t['60%']), int(t['80%']), 1e11]

def test_(u):

    # 获得所有的用户
    d = {'userid': u}
    for col in cols:
        bill_user = bill_data[bill_data.userid==u]
        if col not in ['consume_amount']:
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
    print d
    return d

def multi():
    pool = Pool(12)
    #  按照特征咧进行处理
    users = list(bill_data.userid.unique())
    rst = pool.map(test_,users)
    pool.join()
    pool.close()
    features = pd.DataFrame(rst)

    #    features = pd.merge(features,i,on='userid',how='outer')
    # 效果不错 0.44
    #features.to_csv('../data/train/bill_detail_time_stage5.csv',index=None)


    print features.head()
    print features.shape

    features.to_csv('../data/train/bill_dt_time_5.csv',index=None)

    #features.to_csv('../data/train/bill_detail_time_stage52.csv',index=None)

# 统计 5 个时间段的信用卡数量
def stage_5_bank():
    data = bill_data[['userid','time','bank_id']]
    dd = None
    for i in range(5):
        stg = stage[i]
        di = data[(split_point[i] < (data.time)) & ((data.time) < split_point[i + 1])]
        d = di[['userid','bank_id']]
        d.drop_duplicates(inplace=True)
        d['bank_id'] = d['bank_id'].astype('str')
        #f = d.groupby('userid')['bank_id'].agg(lambda x:":".join(x)).reset_index()
        #f.rename(columns={'bank_id':stg+'bankids'},inplace=True)
        f = d.groupby('userid')['bank_id'].agg(lambda x: len(x)).reset_index()
        f.rename(columns={'bank_id': stg + 'bank_cnt'},inplace=True)
        if i==0:
            dd = f
        else:
            dd = pd.merge(dd,f,how='outer')

    print dd.head()
    print dd.shape
    dd.to_csv("../data/train/bill_stage_bank.csv",index=None)

# 统计放款前后的信用卡数量,以及放款后信用卡新增数量
def time_split_bank():
    data = bill_data[bill_data.time<bill_data.loan_time]
    data = data[['userid','bank_id']]
    data.drop_duplicates(inplace=True)
    f = data.groupby('userid')['bank_id'].agg(lambda x:len(x)).reset_index()
    f.rename(columns={'bank_id':'before_bank_cnt'},inplace=True)

    data = bill_data
    data = data[['userid', 'bank_id']]
    data.drop_duplicates(inplace=True)
    ff = data.groupby('userid')['bank_id'].agg(lambda x: len(x)).reset_index()
    ff.rename(columns={'bank_id': 'all_bank_cnt'}, inplace=True)

    f = pd.merge(f,ff,on='userid',how='outer')
    f['bank_add'] = f['all_bank_cnt'] - f['before_bank_cnt']
    print f.head()
    print f.shape
    f.to_csv("../data/train/bill_all_bank_add.csv",index=None)

# 统计每个阶段的信用卡数量,以及每个阶段新增数量
def stage_m5_bank():
    data = bill_data[['userid','time','bank_id']]
    dd = None
    for i in range(5):
        stg = stage[i]
        di = data[(data.time) < split_point[i + 1]]
        d = di[['userid','bank_id']]
        d.drop_duplicates(inplace=True)
        d['bank_id'] = d['bank_id'].astype('str')
        #f = d.groupby('userid')['bank_id'].agg(lambda x:":".join(x)).reset_index()
        #f.rename(columns={'bank_id':stg+'bankids'},inplace=True)
        f = d.groupby('userid')['bank_id'].agg(lambda x: len(x)).reset_index()
        f.rename(columns={'bank_id': stg + 'mbank_cnt'},inplace=True)
        if i==0:
            dd = f
        else:
            dd = pd.merge(dd,f,how='outer')
    dd['all_mbank_cnt'] = dd['stg5_mbank_cnt'] - dd['stg1_mbank_cnt']
    dd['stg5_mbank_cnt'] = dd['stg5_mbank_cnt'] - dd['stg4_mbank_cnt']
    dd['stg4_mbank_cnt'] = dd['stg4_mbank_cnt'] - dd['stg3_mbank_cnt']
    dd['stg3_mbank_cnt'] = dd['stg3_mbank_cnt'] - dd['stg2_mbank_cnt']
    dd['stg2_mbank_cnt'] = dd['stg2_mbank_cnt'] - dd['stg1_mbank_cnt']
    dd.drop('stg1_mbank_cnt',axis=1,inplace=True)
    print dd.head(100)
    print dd.shape
    dd.to_csv("../data/train/bill_stagem_bank.csv",index=None)


def merge_aa():
    a = pd.read_csv("../data/train/bill_stage_bank.csv")
    b = pd.read_csv("../data/train/bill_all_bank_add.csv")
    c = pd.read_csv("../data/train/bill_stagem_bank.csv")

    d = pd.merge(a,b,on='userid',how='outer')
    d = pd.merge(d,c,on='userid',how='outer')
    print d.head()
    print d.shape
    d.to_csv("../data/train/bill_bank_aaad.csv",index=None)

if __name__=='__main__':
    import warnings
    warnings.simplefilter('ignore')
    #stage_5_bank()
    #time_split_bank()
    #stage_m5_bank()
    merge_aa()












