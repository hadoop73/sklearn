# coding:utf-8



import numpy as np
import pandas as pd
from pandas import DataFrame,Series

import matplotlib.pyplot as plt


"""
   测试数据和训练数据可以一起处理
"""

"""
http://www.csai.cn/yinhang/609440.html
# 循环利息 circ_interest,预借现金额度 prepare_amount,可用金额 avail_amount,消费笔数 consumeN
# 信用卡额度 credit_amount,本期账单余额 amount_of_bill_left,上期还款金额 pre_repayment
# 本期账单金额 amount_of_bill,本期调整金额 adjust_amount

 用户id,账单时间戳,,上期账单金额,
 ,


 用户id,账单时间戳,银行id,上期账单金额,上期还款金额,信用卡额度,
 本期账单余额,本期账单最低还款额,消费笔数,本期账单金额,调整金额,
 循环利息,可用金额,预借现金额度,还款状态

信用额度 credit_amount:信用额度是指信用卡最高可以透支使用的金额

信用卡可用余额 avail_amount:可用余额是你还可以使用的额度，即总额度减去已消费或取现额度再加上你的溢存款。

上期账单金额 pre_amount_of_bill:上期对账单应还的汇总金额

本期余额  amount_of_bill_left:本期余额为负，表明账户有欠款，否则为存款。
补充:若是正数,表示你有消费或提现,需要还款
若是负数,表示有溢缴款,你不需要还款.
若是0,表示你没消费或提现,也无溢缴款,你不需要还款!
　　余额中有“-”号，表示您当前余额有结余;余额中无“-”号，表示您当前的信用卡余额透支。
　　例如：信用卡当前余额为“0”时，表示您的信用卡额度没有透支;信用卡可用余额为“9000”时，
示此卡还有这个额度可透支或消费;假设您的信用卡额度为“9000”，
您透支消费了1000，那么您的当前信用卡余额为“-8000”，信用卡可用余额为“8000”。


上期还款金额 pre_repayment: 从上期账单日到本期账单日期间，您所偿还并且已经入账的还款总额，还款明细列于交易摘要中

本期账单金额 amount_of_bill:从上期账单日到本期账单日期间，各笔交易款项及应收费用的总和，各笔摘要列于交易摘要中

本期调整金额 adjust_amount: 从上期账单日到本期账单日期间，调整交易的金额总和

预借现金 prepare_amount，指持卡人使用信用额度透支取现。
预借现金自银行记账日起收透支利息。
信用卡预借现金额度是指持卡人使用信用卡通过ATM等自助终端提取现金的最高额度。

本期账单金额：又称本期账单应还款金额，指上期账单日到本期帐单日期间各笔交易款项及应收费用的总和，包含用户上期逾期未还款金额。
本期已还金额：指自本期账单日开始至本期到期还款日之间，用户通过各渠道累计还款的金额总和。
本期仍需还款金额：指扣除本期已还金额之后的本期应还款余额。本期仍需还款金额=本期账单金额（本期应还款金额）-本期已还金额。
"""

names = ["userid", "time", "bank_id", "pre_amount_of_bill", "pre_repayment", "credit_amount", \
         "amount_of_bill_left", "least_repayment", "consume_amount", "amount_of_bill", "adjust_amount", \
         "circ_interest", "avail_amount", "prepare_amount", "repayment_state"]

bill_train = pd.read_csv("../../pcredit/train/bill_detail_train.txt", header=None)
bill_test = pd.read_csv("../../pcredit/test/bill_detail_test.txt", header=None)

bill_data = pd.concat([bill_train, bill_test])
bill_data.columns = names


cols = ['pre_amount_of_bill','pre_repayment','credit_amount',
        'amount_of_bill_left','least_repayment','consume_amount',
        'amount_of_bill','circ_interest','avail_amount',
        'prepare_amount']
sts = ['_min','_max','_median','_mean','_std','_cnt','_max_min']


features = pd.DataFrame(columns=['userid'] + [ s+p for p in sts for s in cols])

users = list(bill_data.userid.unique())


def test_(user):
    d = {'userid':user}
    bills = bill_data[bill_data.userid==user]
    for col in cols:
        t = bills[bills[col]!=0][col]
        d[col+'_min'] = t.min()
        d[col + '_max'] = t.max()
        d[col + '_median'] = t.median()
        d[col + '_mean'] = t.mean()
        d[col + '_std'] = t.std()
        d[col + '_cnt'] = t.count()
        d[col + '_max_min'] = t.max() - t.min()
    r = pd.DataFrame(d,index=[0])
    return r


from multiprocessing import Pool
pool = Pool(12)

res = []
for u in users:
    res.append(pool.apply_async(test_,args=(u,)))

pool.close()
pool.join()

rst = [i.get() for i in res]

for i in rst:
    features = pd.concat([features,i],axis=0)
features.to_csv('../data/train/bill_detail.csv',index=None)
#  获得


"""

#  1)获取消费笔数 ,消费笔数从 sum 改成 mean
consume_amount = bill_data[['userid','consume_amount']]
consume_amount = consume_amount[consume_amount['consume_amount']!=0]  # 只选不为 0 的数

#  获取信用卡一期单笔消费笔数的最大值
consume_amount_max = pd.pivot_table(consume_amount,index=['userid'],values=['consume_amount'],aggfunc=np.max)
consume_amount_max.columns = ['consume_amount_max']


consume_amount_mean = pd.pivot_table(consume_amount,index=['userid'],values=['consume_amount'],aggfunc=np.mean)
consume_amount_mean.columns = ['consume_amount_mean']

consume_amount_sum = pd.pivot_table(consume_amount,index=['userid'],values=['consume_amount'],aggfunc=np.sum)
consume_amount_sum.columns = ['consume_amount_sum']

DATAS = consume_amount_sum.join([consume_amount_max,consume_amount_mean])


#  消费笔数，信用卡额度 credit_amount，本期余额 amount_of_bill_left，上期还款金额 pre_repayment    本期账单金额 amount_of_bill
#  获取最大值，最小值，每张卡的平均值的最小，大值  
# pre_amount_of_bill	pre_repayment	credit_amount	amount_of_bill_left	least_repayment
# amount_of_bill	adjust_amount	circ_interest	avail_amount	prepare_amount
cols = ['pre_amount_of_bill','pre_repayment','credit_amount',
        'amount_of_bill_left','least_repayment','amount_of_bill','avail_amount','prepare_amount']

for col in cols:
    datas = bill_data[['userid',col]]
    datas = datas[datas[col]!=0]  # 只选不为 0 的数
    data_max = pd.pivot_table(datas,index=['userid'],values=[col],aggfunc=np.max)  # 获取最大值
    data_max.columns = [ "{}_max".format(col) ]
    data_min = pd.pivot_table(datas,index=['userid'],values=[col],aggfunc=np.min)  # 获取最小值
    data_min.columns = [ "{}_min".format(col) ]

    data_mean = pd.pivot_table(datas, index=['userid'], values=[col], aggfunc=np.mean)  # 获取平均值
    data_mean.columns = ["{}_mean".format(col)]

    DATAS = DATAS.join([data_max,data_min])
    #  最大值和最小值的差
    DATAS['{}_sub'.format(col)] = DATAS["{}_max".format(col) ] - DATAS["{}_min".format(col) ]

#  统计信用卡数目
t = bill_data[['userid','bank_id']]
t['bank_n'] = 1
data = t.groupby(['userid','bank_id'])['bank_n'].agg(lambda x: 1)
data = data.sum(level='userid')
DATAS = DATAS.join(data)

#  按照信用卡分组统计消费总数，平均值
consume_amount_bank = bill_data[['userid','bank_id','consume_amount']]
consume_amount_bank = consume_amount_bank[consume_amount_bank['consume_amount']!=0]  # 只选不为 0 的数
consume_amount_bank_sum = pd.pivot_table(consume_amount_bank,index=['userid','bank_id'],values=['consume_amount'],aggfunc=np.sum)
consume_amount_bank_sum.columns = ['consume_amount_bank_sum']

consume_amount_bank_sum_max = consume_amount_bank_sum.max(level='userid')  #  获得最大值
consume_amount_bank_sum_max.columns = ['consume_amount_bank_sum_max']

consume_amount_bank_sum_min = consume_amount_bank_sum.min(level='userid')  #  获得最小值
consume_amount_bank_sum_min.columns = ['consume_amount_bank_sum_min']

consume_amount_bank_sum_mean = consume_amount_bank_sum.mean(level='userid')  #  获得平均值
consume_amount_bank_sum_mean.columns = ['consume_amount_bank_sum_mean']

consume_amount_bank_sum_min_max = consume_amount_bank_sum_min.join([consume_amount_bank_sum_max,consume_amount_bank_sum_mean])

DATAS = DATAS.join(consume_amount_bank_sum_min_max)

#  每张卡的平均值的最大值
consume_amount_bank = bill_data[['userid','bank_id','consume_amount']]
consume_amount_bank = consume_amount_bank[consume_amount_bank['consume_amount']!=0]  # 只选不为 0 的数
consume_amount_bank_mean = pd.pivot_table(consume_amount_bank,index=['userid','bank_id'],values=['consume_amount'],aggfunc=np.mean)
consume_amount_bank_mean.columns = ['consume_amount_bank']

consume_amount_bank_mean_max = consume_amount_bank_mean.max(level='userid')
consume_amount_bank_mean_max.columns = ['consume_amount_bank_mean_max']

consume_amount_bank_mean_min = consume_amount_bank_mean.min(level='userid')
consume_amount_bank_mean_min.columns = ['consume_amount_bank_mean_min']
consume_amount_bank_mean_max_min = consume_amount_bank_mean_max.join(consume_amount_bank_mean_min)

DATAS = DATAS.join(consume_amount_bank_mean_max_min)

#    每张卡 消费笔数，信用卡额度 credit_amount，本期余额 amount_of_bill_left，上期还款金额 pre_repayment   本期账单金额 amount_of_bill  本期账单最低还款额 least_repayment
#  获取最大值
data_bank_max_min = consume_amount_bank_mean_max_min.copy()
#cols = ['credit_amount', 'amount_of_bill_left', 'pre_repayment', 'amount_of_bill']
for col in cols:
    datas = bill_data[['userid', 'bank_id', col]]
    datas = datas[datas[col] != 0]  # 只选不为 0 的数

    datas_mean = pd.pivot_table(datas, index=['userid', 'bank_id'], values=[col], aggfunc=np.mean)

    data_bank_mean_max = datas_mean.max(level='userid')
    data_bank_mean_max.columns = ["{}_bank_mean_max".format(col)]

    data_bank_mean_min = datas_mean.min(level='userid')
    data_bank_mean_min.columns = ["{}_bank_mean_min".format(col)]

    DATAS = DATAS.join([data_bank_mean_max, data_bank_mean_min])

DATAS['bill_tag'] = 1

DATAS.to_csv('../data/train/bill_detail.csv')

print "datas size: ",DATAS.shape
print "data writed in file: ",'../data/train/bill_detail.csv'

"""







