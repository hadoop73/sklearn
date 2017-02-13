# coding:utf-8


import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn import metrics



def bag():
    f1 = pd.read_csv('../data/result_xgk_821.csv')
    f2 = pd.read_csv('../data/result_xgk_634.csv')
    f3 = pd.read_csv('../data/result_xgk_225.csv')
    d = f1
    d.probability = (f1.probability + f2.probability + f3.probability)/3
    d.to_csv('../data/result_bg1.csv',index=None)


if __name__=='__main__':
    bag()

