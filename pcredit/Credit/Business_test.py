import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt

os.chdir('/Users/Evan/Kaggle/')

user_info_test = pd.read_csv('./Credit/test/user_info_test.txt',names=['ID', 'gender','career','education', 'marriage', 'hukou'],header=None)
loan_time_test = pd.read_csv('./Credit/test/loan_time_test.txt',names=['ID', 'timestamp_money'], header=None)
usersID_test = pd.read_csv('./Credit/test/usersID_test.txt',names=['ID','Label'],header=None)
bank_detail_test = pd.read_csv('./Credit/test/bank_detail_test.txt',names=['ID', 'timestamp_bank', 'type', 'money','salary'],header=None)
browse_history_test = pd.read_csv('./Credit/test/browse_history_test.txt',names=['ID', 'timestamp_browse', 'browse_data', 'browser_code'],header=None)
bill_detail_test = pd.read_csv('./Credit/test/bill_detail_test.txt',names=['ID', 'timestamp_bill','bank_id', 'last_bill', 'last_repayment', 'credit_line', 'current_balance', 'minimum_payments', 'amount_transactions', 'current_money', 'adjust_money','cycle_interest','available_money','cash_advance','repayment_status'],header=None)

data_user_test = pd.merge(user_info_test,pd.merge(loan_time_test,usersID_test,on='ID',how='inner'),on='ID',how='inner')
data_user_test = data_user_test.apply(lambda x: x.astype(str))

data_user_test.duplicated().sum()
data_user_test.isnull().sum()

# Bank information extraction

# 净收入

bank_detail_test['money_direction'] = bank_detail_test['type'].replace({0:1,1:-1})*bank_detail_test['money']
data_net = bank_detail_test.groupby(['ID'])['money_direction'].agg({'sum','count','mean'}).rename(columns=dict(sum='sum_total',count='freq_total',mean='mean_total'))
data_net['ID'] = data_net.index
data_net.reset_index(drop = True)
data_net['ID'] = data_net['ID'].apply(lambda x:str(x))

# 收入信息汇总

data_income = bank_detail_test[bank_detail_test.type==0].groupby(['ID'])['money'].agg({'sum','count','mean'}).rename(columns=dict(sum='sum_income', count='freq_income',mean='mean_income'))
data_income['ID'] = data_income.index
data_income.reset_index(drop = True)
data_income['ID'] = data_income['ID'].apply(lambda x:str(x))

bank_detail_test.groupby("ID")['ID'].count().to_frame().shape

len(data_income)/len(data_user_test)

# 支出信息汇总

data_spend = bank_detail_test[bank_detail_test.type==1].groupby(['ID'])['money'].agg({'sum','count','mean'}).rename(columns=dict(sum='sum_spend', count='freq_spend',mean='mean_spend'))
data_spend['ID'] = data_spend.index
data_spend.reset_index(drop = True)
data_spend['ID'] = data_spend['ID'].apply(lambda x:str(x))

data_bank = pd.merge(pd.merge(pd.merge(data_user_test,data_net,on='ID',how='left'),data_income,on='ID',how='left'),data_spend,on='ID',how='left')

data_bank.head()

# Browser information extraction

data_browse = browse_history_test.loc[:, ['ID', 'browse_data']].groupby(['ID']).mean()
data_browse.head()

# Bill information extraction

print(bill_detail_test.columns)
print(bank_detail_test.columns)
bill_detail_test.head()

data_bill = bill_detail_test.groupby(['ID'])['credit_line','cash_advance','amount_transactions'].sum()
bill_detail_test.groupby(['ID'])['credit_line','cash_advance','amount_transactions'].sum().head()
bill_detail_test.groupby(['ID'])['credit_line','cash_advance','amount_transactions'].sum().reset_index().head()
bill_detail_test.groupby(['ID'])['credit_line','cash_advance','amount_transactions'].sum().head()
data_bill = bill_detail_test.assign(bill_credit=bill_detail_test.last_bill/bill_detail_test.credit_line).groupby('ID')['bill_credit'].mean().to_frame().replace({np.inf:2}).reset_index().merge(bill_detail_test.groupby(['ID'])['credit_line','cash_advance','amount_transactions'].sum().reset_index(),on='ID',how='outer')

data_bill.head()

data_browser = browse_history_test.groupby(['ID','browser_code']).browse_data.sum().reset_index().pivot(index='ID', columns='browser_code', values='browse_data').fillna(0)

data_browser.columns = ['browser'+ str(x) for x in data_browser.columns]

data_browser.reset_index().head()

# 数据合并

data_bill.ID = data_bill.ID.astype(str)
data_browser = data_browser.reset_index()
data_browser.ID = data_browser.ID.astype(str)

data_test = data_bank.merge(data_bill,on='ID',how='left').merge(data_browser,on='ID',how='left')

print(data_test.shape)

print(data_test.isnull().sum()/len(data_test))

print(data_test.shape)

data_test.to_csv('/Users/Evan/Desktop/data_test.csv')

pd.crosstab(data_user_test.gender,data_user_test.Label,margins=True)
pd.crosstab(data_user_test.career,data_user_test.Label).apply(lambda x: x/x.sum(), axis=1)
pd.crosstab(data_user_test.gender,data_user_test.Label).apply(lambda x: x/x.sum(), axis=1)
pd.crosstab(data_user_test.education,data_user_test.Label).apply(lambda x: x/x.sum(), axis=1)
pd.crosstab(data_user_test.marriage,data_user_test.Label).apply(lambda x: x/x.sum(), axis=1)
pd.crosstab(data_user_test.hukou,data_user_test.Label).apply(lambda x: x/x.sum(), axis=1)

bank_detail_test['money_direction'] = bank_detail_test['type'].replace({0:1,1:-1})*bank_detail_test['money']
data_net = bank_detail_test.groupby(['ID'])['money_direction'].agg({'sum','count','mean'}).rename(columns=dict(sum='sum_total',count='freq_total',mean='mean_total'))
data_net['ID'] = data_net.index
data_net.reset_index(drop = True)
data_net['ID'] = data_net['ID'].apply(lambda x:str(x))

data_income = bank_detail_test[bank_detail_test.type==0].groupby(['ID'])['money'].agg({'sum','count','mean'}).rename(columns=dict(sum='sum_income', count='freq_income',mean='mean_income'))
data_income['ID'] = data_income.index
data_income.reset_index(drop = True)
data_income['ID'] = data_income['ID'].apply(lambda x:str(x))

bank_detail_test.groupby("ID")['ID'].count().to_frame().shape

len(data_income)/len(data_user_test)



data_spend = bank_detail_test[bank_detail_test.type==1].groupby(['ID'])['money'].agg({'sum','count','mean'}).rename(columns=dict(sum='sum_spend', count='freq_spend',mean='mean_spend'))
data_spend['ID'] = data_spend.index
data_spend.reset_index(drop = True)
data_spend['ID'] = data_spend['ID'].apply(lambda x:str(x))

data_bank = pd.merge(pd.merge(pd.merge(data_user_test,data_net,on='ID',how='left'),data_income,on='ID',how='left'),data_spend,on='ID',how='left')
data_bank.head()


data_browse = browse_history_test.loc[:, ['ID', 'browse_data']].groupby(['ID']).mean()
data_browse.head()


print(bill_detail_test.columns)
print(bank_detail_test.columns)
bill_detail_test.head()

data_bill = bill_detail_test.groupby(['ID'])['credit_line','cash_advance','amount_transactions'].sum()
bill_detail_test.groupby(['ID'])['credit_line','cash_advance','amount_transactions'].sum().head()
bill_detail_test.groupby(['ID'])['credit_line','cash_advance','amount_transactions'].sum().reset_index().head()
bill_detail_test.groupby(['ID'])['credit_line','cash_advance','amount_transactions'].sum().head()
data_bill = bill_detail_test.assign(bill_credit=bill_detail_test.last_bill/bill_detail_test.credit_line).groupby('ID')['bill_credit'].mean().to_frame().replace({np.inf:2}).reset_index().merge(bill_detail_test.groupby(['ID'])['credit_line','cash_advance','amount_transactions'].sum().reset_index(),on='ID',how='outer')

data_bill.head()

data_browser = browse_history_test.groupby(['ID','browser_code']).browse_data.sum().reset_index().pivot(index='ID', columns='browser_code', values='browse_data').fillna(0)

data_browser.columns = ['browser'+ str(x) for x in data_browser.columns]

data_browser.reset_index().head()

data_bill.ID = data_bill.ID.astype(str)
data_browser = data_browser.reset_index()
data_browser.ID = data_browser.ID.astype(str)

data_test = data_bank.merge(data_bill,on='ID',how='left').merge(data_browser,on='ID',how='left')

print(data_test.shape)
print(data_test.isnull().sum()/len(data_test))
print(data_test.shape)

data_test.to_csv('/Users/Evan/Desktop/data_test.csv')
