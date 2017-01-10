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


def overdue():
    # overdue_train，这是我们模型所要拟合的目标
    target = pd.read_csv('../../pcredit/train/overdue_train.txt',
                         header=None)
    target.columns = ['userid', 'label']
    target.index = target['userid']
    target.drop('userid',
                axis=1,
                inplace=True)
    return target

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

overdue_data = overdue()

train = user_train.join([bank_train,bill_train,browser_train,overdue_data])
test = user_test.join([bank_test,bill_test,browser_test])

print 'train size:',train.shape
bank_train_null = train[train['bank'].isnull()]
bill_train_null = train[train['bill'].isnull()]
browser_train_null = train[train['browser'].isnull()]

overdue_bank_null = train[train['label']==1]
print "\noverdue size:",overdue_bank_null.shape


overdue_bank_null = train[(train['label']==1) & train['bank'].isnull()]
overdue_bill_null = train[(train['label']==1) & train['bill'].isnull()]
overdue_browser_null = train[(train['label']==1) & train['browser'].isnull()]

print "\noverdue bank null:",overdue_bank_null.shape
print "overdue bill null:",overdue_bill_null.shape
print "overdue browser null:",overdue_browser_null.shape

overdue_bank_null = train[(train['label']==1) & train['bank'].notnull()]
overdue_bill_null = train[(train['label']==1) & train['bill'].notnull()]
overdue_browser_null = train[(train['label']==1) & train['browser'].notnull()]

print "\noverdue bank not null:",overdue_bank_null.shape
print "overdue bill not null:",overdue_bill_null.shape
print "overdue browser not null:",overdue_browser_null.shape

overdue_bank_null = train[(train['label']==0) & train['bank'].isnull()]
overdue_bill_null = train[(train['label']==0) & train['bill'].isnull()]
overdue_browser_null = train[(train['label']==0) & train['browser'].isnull()]

print "\nnot overdue bank null:",overdue_bank_null.shape
print "not overdue bill null:",overdue_bill_null.shape
print "not overdue browser null:",overdue_browser_null.shape

print "\nbank train null:",bank_train_null.shape
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
train size: (55596, 9)

overdue size: (7183, 9)

overdue bank null: (5723, 9)
overdue bill null: (578, 9)
overdue browser null: (987, 9)

overdue bank not null: (1460, 9)
overdue bill not null: (6605, 9)
overdue browser not null: (6196, 9)

not overdue bank null: (40579, 9)
not overdue bill null: (1844, 9)
not overdue browser null: (7279, 9)

bank train null: (46302, 9)
bill train null: (2422, 9)
browser train null: (8266, 9)

bank bill train null: (416, 9)
bill browser train null: (324, 9)
browser bank train null: (7194, 9)
all train null: (37, 9)

test size: (13899, 8)
bank test null: (13190, 8)
bill test null: (256, 8)
browser test null: (1902, 8)

bank bill test null: (60, 8)
bill browser test null: (11, 8)
browser bank test null: (1871, 8)
all test null: (2, 8)
'''

