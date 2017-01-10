# coding:utf-8



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def merge():
    user_info = pd.read_csv("../data/train/user_info_train.csv")
    bank_data = pd.read_csv("../data/train/bank_data.csv")
    bill_data = pd.read_csv("../data/train/bill_detail.csv")
    browse_data = pd.read_csv("../data/train/browse_history.csv")


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
    datas.to_csv('../data/train_data_7.csv')
    print datas.head()


if __name__=='__main__':
    merge()


























