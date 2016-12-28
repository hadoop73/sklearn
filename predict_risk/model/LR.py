# coding:utf-8

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report



from GetData import getDatas
from WriteDatas import writeDatas

#  都是 pandas 的 DataFrame

train,target,test = getDatas()


# 这里用Logistic回归
clf = LogisticRegression(C = 1.0,
                              penalty = 'l2')


from ROC import ROC

clf = ROC(clf,train,target)




















