# coding:utf-8



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


user_info = pd.read_csv("../data/train/user_info_dummy.csv")
bank_data = pd.read_csv("../data/train/bank_dummy_data.csv")
bill_data = pd.read_csv("../data/train/bill_dummy_data.csv")


datas = user_info.join([bank_data,bill_data])

datas.fillna(0)
print datas.head()




























