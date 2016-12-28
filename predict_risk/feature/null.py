# coding:utf-8


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 训练 用户id,性别,职业,教育程度,婚姻状态,户口类型
names = ['userid','sex','job','edu','marriage','account']
user_info_train = pd.read_csv("../../pcredit/train/user_info_train.txt",header=None)
user_info_train.columns = names
user_train = pd.pivot_table(user_info_train,index=["userid"],values=names)

# 测试 用户id,性别,职业,教育程度,婚姻状态,户口类型
user_info_test = pd.read_csv("../../pcredit/test/user_info_test.txt",header=None)
user_info_test.columns = names
user_test = pd.pivot_table(user_info_test,index=["userid"],values=names)

# 返回训练数据,测试数据
def bank_data_an():
    #  训练 银行 数据
    names = ['userid','time','extype','examount','mark']
    bank_detail_train = pd.read_csv("../../pcredit/train/bank_detail_train.txt",header=None)
    bank_detail_train.columns = names
    #  测试数据集 银行
    bank_detail_test = pd.read_csv("../../pcredit/test/bank_detail_test.txt",header=None)
    bank_detail_test.columns = names

    #  缺失用户数分析
    ##  1)统计收入 examount 的均值  2) 统计支出 examount  的均值
    amount_data = pd.pivot_table(bank_detail_train,index=['userid'],values=['examount'])
    amount_data.columns = ['bank']
    ##  1)统计收入 examount 的均值  2) 统计支出 examount  的均值
    amount_test_data = pd.pivot_table(bank_detail_test,index=['userid'],values=['examount'])
    amount_test_data.columns = ['bank']
    return amount_data,amount_test_data

# bill 数据
def bill_data_an():
    names = ["userid", "time", "bank_id", "pre_amount_of_bill", "pre_repayment", "credit_amount", \
                 "amount_of_bill_left", "least_repayment", "consume_amount", "amount_of_bill", "adjust_amount", \
                 "circ_interest", "avail_amount", "prepare_amount", "repayment_state"]
    bill_train = pd.read_csv("../../pcredit/train/bill_detail_train.txt", header=None)
    bill_test = pd.read_csv("../../pcredit/test/bill_detail_test.txt", header=None)

    bill_train.columns = names
    bill_test.columns=names
    bill_users = pd.pivot_table(bill_train,index=['userid'],values=['consume_amount'])
    bill_test_users = pd.pivot_table(bill_test,index=['userid'],values=['consume_amount'])
    bill_users.columns = ['bill']
    bill_test_users.columns = ['bill']
    return bill_users,bill_test_users


# 浏览记录
def browser_data_an():
    names = ['userid','time','browser_behavior','browser_behavior_number']
    browse_history_train = pd.read_csv("../../pcredit/train/browse_history_train.txt",header=None)
    browse_history_train.columns = names
    browse_history_test = pd.read_csv("../../pcredit/test/browse_history_test.txt",header=None)
    browse_history_test.columns = names
    browse_user = pd.pivot_table(browse_history_train,index=['userid'],values=['browser_behavior_number'])
    browse_test_user = pd.pivot_table(browse_history_test,index=['userid'],values=['browser_behavior_number'])
    browse_user.columns = ['browser']
    browse_test_user.columns = ['browser']
    return browse_user,browse_test_user


bank_train,bank_test = bank_data_an()
bill_train,bill_test = bill_data_an()
browser_train,browser_test = browser_data_an()

train = user_train.join([bank_train,bill_train,browser_train])
test = user_test.join([bank_test,bill_test,browser_test])

print 'train size:',train.shape
bank_train_null = train[train['bank'].isnull()]
bill_train_null = train[train['bill'].isnull()]
browser_train_null = train[train['browser'].isnull()]

print "bank train null:",bank_train_null.shape
print "bill train null:",bill_train_null.shape
print "browser train null:",browser_train_null.shape

bank_bill_train_null = train[(train['bank'].isnull() & train['bill'].isnull())]
bill_browser_train_null = train[(train['bill'].isnull() & train['browser'].isnull())]
browser_bank_train_null = train[(train['browser'].isnull() & train['bank'].isnull())]

all_train_null = train[(train['browser'].isnull() & train['bill'].isnull() & train['bank'].isnull())]

print "\nbank bill train null:",bank_bill_train_null.shape
print "bill browser train null:",bill_browser_train_null.shape
print "browser bank train null:",browser_bank_train_null.shape
print "all train null:",all_train_null.shape

print '\ntest size:',test.shape
bank_test_null = test[test['bank'].isnull()]
bill_test_null = test[test['bill'].isnull()]
browser_test_null = test[test['browser'].isnull()]

print "bank test null:",bank_test_null.shape
print "bill test null:",bill_test_null.shape
print "browser test null:",browser_test_null.shape

bank_bill_test_null = test[(test['bank'].isnull() & test['bill'].isnull())]
bill_browser_test_null = test[(test['bill'].isnull() & test['browser'].isnull())]
browser_bank_test_null = test[(test['browser'].isnull() & test['bank'].isnull())]

all_test_null = test[(test['browser'].isnull() & test['bill'].isnull() & test['bank'].isnull())]

print "\nbank bill test null:",bank_bill_test_null.shape
print "bill browser test null:",bill_browser_test_null.shape
print "browser bank test null:",browser_bank_test_null.shape
print "all test null:",all_test_null.shape

'''
train size: (55596, 8)
bank train null: (46302, 8)
bill train null: (2422, 8)
browser train null: (8266, 8)

bank bill train null: (416, 8)
bill browser train null: (324, 8)
browser bank train null: (7194, 8)
all train null: (37, 8)

test size: (13899, 8)
bank test null: (13190, 8)
bill test null: (256, 8)
browser test null: (1902, 8)

bank bill test null: (60, 8)
bill browser test null: (11, 8)
browser bank test null: (1871, 8)
all test null: (2, 8)
'''

