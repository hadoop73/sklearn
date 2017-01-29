# coding:utf-8



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def merge(user='',bill=''):
    # user_info_dummy   user_info_train
    file_user = "../data/train/user_info_data{}.csv".format(user)
    file_bank = "../data/train/bank_detail.csv"
    file_bill = "../data/train/bill_detail10{}.csv".format(bill)
    file_browser = "../data/train/browse_history.csv"
    user_info = pd.read_csv(file_user)
    bank_data = pd.read_csv(file_bank)
    bill_data = pd.read_csv(file_bill)
    browse_data = pd.read_csv(file_browser)

    """
    print user_info.head()
    print bank_data.head()
    print bill_data.head()
    print browse_data.head()
    """

    print "merge datas:"
    print "\t",file_user
    print "\t",file_bank
    print "\t",file_bill
    print "\t",file_browser

    user_info = user_info.set_index('userid')
    bank_data = bank_data.set_index('userid')
    bill_data = bill_data.set_index('userid')
    browse_data = browse_data.set_index('userid')

    print bill_data.head()
    datas = user_info.join([bank_data,bill_data,browse_data])
    #print datas.head()
    datas = datas.fillna(0)
    file_result = '../data/train_data_0{}.csv'.format(bill)
    print "datas produced!"
    print "\t",file_result
    print "datas size: ", datas.shape
    #datas.index =datas.index.astype(int)
    datas.to_csv(file_result)
    print datas.head()


#from predict_risk.model.GBDT import gbdt

if __name__=='__main__':

    merge(bill="")
    print "merge finished!!!!"


























