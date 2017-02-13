# coding:utf-8



import numpy as np
import pandas as pd
from pandas import DataFrame,Series
import seaborn as sns
import matplotlib.pyplot as plt


def get_bill_data():

    names = ["userid", "time", "bank_id", "pre_amount_of_bill", "pre_repayment", "credit_amount", \
             "amount_of_bill_left", "least_repayment", "consume_amount", "amount_of_bill", "adjust_amount", \
             "circ_interest", "avail_amount", "prepare_amount", "repayment_state"]

    bill_train = pd.read_csv("../../pcredit/train/bill_detail_train.txt", header=None)
    bill_test = pd.read_csv("../../pcredit/test/bill_detail_test.txt", header=None)
    bill_data = pd.concat([bill_train, bill_test])
    bill_data.columns = names
    del bill_train, bill_test
    return bill_data




"""
bill 的重复数据太多了，需要删除
63	1265	0	7	20.074529	18.233999	19.748127	19.917975	17.751187	1	15.792805	0.0	0.0	0.0	19.748127	0
74	1265	0	7	20.074529	18.233999	19.748127	19.917975	17.751187	1	15.792805	0.0	0.0	0.0	0.000000	0
(2753013, 15)
(1453798, 15) 去掉重复后的数据,时间相同可以相同，后面的数据不同

(2053516, 15)  去掉重复数据，包括时间相同的数据

(1741095, 15)  时间不等于 0 的数据，时间不同的数据

(471009, 15) 时间=0的数据
"""

def delete_duplicate(bill_data):

    bill_data = bill_data[bill_data.time!=0]

    bill_data.drop_duplicates(['userid','time','bank_id','pre_amount_of_bill','pre_repayment'],inplace=True)


    print bill_data.head()
    print bill_data.shape

    bill_data.to_csv('data/train/bill_diff.csv',index=None)

def delete_duplicate_with0(bill_data):
    """
    没有删除 时间为 0 的记录，还剩 (2053516, 15)
    :param bill_data:
    :return:
    """

    bill_data.drop_duplicates(['userid','time','bank_id','pre_amount_of_bill','pre_repayment'],inplace=True)


    print bill_data.head()
    print bill_data.shape

    bill_data.to_csv('data/train/bill_diff_with0.csv',index=None)


def record_cnt():
    d = pd.read_csv('data/train/bill_diff_with0.csv')
    d['cnt'] = 1
    a = d[['userid','cnt']].groupby('userid').sum().reset_index()

    target = pd.read_csv('../../pcredit/train/overdue_train.txt',
                         header=None)

    target.columns = ['userid', 'label']

    a = pd.merge(a,target,on='userid',how='left')
    sns.kdeplot(a[a.label==1]['cnt'],color='r')
    sns.kdeplot(a[a.label == 0]['cnt'], color='b')
    plt.title('user wich n records')
    plt.show()

# 统计每张信用卡记录数量

def cnt_record():
    bill_data = pd.read_csv('data/train/bill_diff.csv')
    bill_data.drop_duplicates(['userid', 'bank_id', 'pre_amount_of_bill', 'pre_repayment'], inplace=True)
    t = bill_data[['userid','bank_id']]
    t['cnt'] = 1
    t = t.groupby(['userid','bank_id'])['cnt'].sum().reset_index()
    print t['cnt'].sort_values(ascending=False)
    sns.kdeplot(t[t['cnt']<=24]['cnt'], color='r')
    plt.title('cnt lt 24')
    plt.show()
    print t.head()
    print t.shape

def cnt_record2():
    """
    按照 user 统计记录条数
    (64059, 2)  总数
    (57724, 2) 小于 50 90%
    (62647, 2) 0.977957820135 小鱼 100
    (63701, 2) 0.994411401989 小于 150
    (35425, 2) 0.553005822757 12
    (24613, 2) 0.384223918575 6
    :return:
    """
    bill_data = pd.read_csv('data/train/bill_diff.csv')
    bill_data.drop_duplicates(['userid', 'bank_id', 'pre_amount_of_bill', 'pre_repayment'], inplace=True)
    t = bill_data[['userid']]
    t['cnt'] = 1
    t = t.groupby(['userid'])['cnt'].sum().reset_index()
    print t['cnt'].sort_values(ascending=False)
    t = t[t['cnt']<=6]
    #sns.kdeplot(t['cnt']<100, color='r')
    #plt.title('cnt per user lt 50')
    #plt.show()
    print t.head()
    print t.shape,t.shape[0]*1.0/64059



if __name__=='__main__':
    #bill = get_bill_data()
    #delete_duplicate_with0(bill)
    record_cnt()




