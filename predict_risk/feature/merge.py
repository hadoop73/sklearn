# coding:utf-8



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def merge(user='',bill=''):
    file_user = "../data/train/user_info_train{}.csv".format(user)
    file_bank = "../data/train/bank_detail.csv"
    file_bill = "../data/train/bill_detail{}.csv".format(bill)
    file_browser = "../data/train/browse_history.csv"
    user_info = pd.read_csv(file_user)
    bank_data = pd.read_csv(file_bank)
    bill_data = pd.read_csv(file_bill)
    browse_data = pd.read_csv(file_browser)

    print "merge datas:"
    print "\t",file_user
    print "\t",file_bank
    print "\t",file_bill
    print "\t",file_browser

    user_info.index = user_info['userid']
    user_info.drop(['userid'],axis=1,inplace=True)

    bank_data.index = bank_data['userid']
    bank_data.drop(['userid'],axis=1,inplace=True)

    bill_data.index = bill_data['userid']
    bill_data.drop(['userid'],axis=1,inplace=True)


    browse_data.index = browse_data['userid']
    browse_data.drop(['userid'],axis=1,inplace=True)

    datas = user_info.join([bank_data,bill_data,browse_data],how='outer')

    datas = datas.fillna(0)
    file_result = '../data/train_data_{}.csv'.format(bill)
    print "datas produced!"
    print "\t",file_result
    print "datas size: ", datas.shape
    datas.to_csv(file_result)
    print datas.head()


from predict_risk.model.GBDT import gbdt

if __name__=='__main__':

    merge(bill="")
    print "merge finished!!!!"


























