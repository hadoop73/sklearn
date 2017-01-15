#  coding:utf-8


import re
import math
import collections
import numpy as np
import time
import operator
from scipy.io import mmread, mmwrite
from random import randint
from sklearn import cross_validation
from sklearn import linear_model
from sklearn.grid_search import GridSearchCV
from sklearn import preprocessing as pp
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble import  RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import ExtraTreesRegressor
#from sklearn.decomposition import ProbabilisticPCA, KernelPCA
from sklearn.decomposition import NMF
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
import scipy.stats as stats
from sklearn import tree
from sklearn.feature_selection import f_regression
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc, f1_score
from sklearn.gaussian_process import GaussianProcess


from GetData import getDatas
from WriteDatas import writeDatas

import pandas as pd

train,target,test = getDatas("train_data_")



f_val, p_val = f_regression(train,target)

f_val_dict = {}
p_val_dict = {}
for i in range(len(f_val)):
    if math.isnan(f_val[i]):
        f_val[i] = 0.0
    f_val_dict[i] = f_val[i]
    if math.isnan(p_val[i]):
        p_val[i] = 0.0
    p_val_dict[i] = p_val[i]

# operator.itemgetter
# a = [1,2,3]
# >>> b=operator.itemgetter(1)      //定义函数b，获取对象的第1个域的值
# >>> b(a)
# 2
# 这里获得第二个值，也就是 value 的排序
sorted_f = sorted(f_val_dict.iteritems(), key=operator.itemgetter(1), reverse=True)  # 定义函数b，获取对象的第1个域的值
sorted_p = sorted(p_val_dict.iteritems(), key=operator.itemgetter(1), reverse=True)

#  选取排名靠前 的 n 个特征

cols = train.columns
n_features = len(cols) - 5

feature_indexs = []
for i in range(0,n_features):
    feature_indexs.append(cols[sorted_f[i][0]])

#print feature_indexs
#return feature_indexs

def  gbm(train,test,target,alpha):
    sub_x_Train = train[feature_indexs]

    gbc = GradientBoostingClassifier(n_estimators=500, max_depth=5)
    gbc.fit(sub_x_Train, target)

    sub_x_Test = test[feature_indexs]

    pred_probs = gbc.predict_proba(sub_x_Test)[:, 1]  # 获得预测的概率值

    ind_test = np.where(pred_probs > alpha)[0]  # 因为 pred_probs 是列向量， 获得概率大于 0.55 的行


    ind_train = np.where(target > 0)[0]  # 获得训练数据为 1 的行

    #  标准化训练数据

    scaler = pp.StandardScaler()
    scaler.fit(sub_x_Train)
    sub_x_Train = scaler.transform(sub_x_Train)
    sub_x_Test = scaler.transform(sub_x_Test)

    #print "sub_x_Train",sub_x_Train
    #print "sub_x_Test", sub_x_Test

    # 再用 GBDT 做回归预测
    gbr1000 = GradientBoostingRegressor(n_estimators=500, max_depth=4, subsample=0.5, learning_rate=0.05)

    gbr1000.fit(sub_x_Train[ind_train],target.iloc[ind_train])

    preds_all = np.zeros([len(sub_x_Test)])

    preds = gbr1000.predict(sub_x_Test[ind_test])
    preds_all[ind_test] =  preds  # 使用 e^p 作为预测值

    return preds_all

    #writeDatas(preds_all, test, "{}".format(9))


from sklearn import metrics
from sklearn.cross_validation import train_test_split


ind_train = np.where(target > 0)[0]  # 获得训练数据为 1 的行



label = target.iloc[ind_train]
#print label

trainX = train.iloc[ind_train]

ind_train0 = np.where(target == 0)[0]  # 获得训练数据为 1 的行
label0 = target.iloc[ind_train0]
trainX0 = train.iloc[ind_train0]

train_X, test_X, train_y, test_y = train_test_split(trainX,
                                                    label,
                                                    test_size=0.2,
                                                    random_state=0)

train_X0, test_X0, train_y0, test_y0 = train_test_split(trainX0,
                                                    label0,
                                                    test_size=0.2,
                                                    random_state=0)



train_X = pd.concat([train_X,train_X0])
train_X = train_X.sort_index()

test_X = pd.concat([test_X,test_X0])
test_X = test_X.sort_index()

train_y = pd.concat([train_y,train_y0])
train_y = train_y.sort_index()

test_y = pd.concat([test_y,test_y0])
test_y = test_y.sort_index()


for alpha in [1.5,0.2,0.18,0]:
    try:
        scores = gbm(train_X,test_X,train_y,alpha)
        fp, tp, thresholds = metrics.roc_curve(test_y, scores, pos_label=1)
        ks = np.max(tp - fp)
        print "{}K-S:{}".format(alpha,ks)
        print "AUC:{}".format(metrics.auc(fp, tp))
    except :
        continue








