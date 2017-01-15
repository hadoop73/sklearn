# coding:utf-8

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cross_validation import train_test_split

"""
只适用于参数设置好的模型
fit 用于训练
predict 用于预测
"""

import math

def  ROC(model,train,target):

    #train.drop(['userid'], axis=1, inplace=True)
    train_X, test_X, train_y, test_y = train_test_split(train,
                                                        target.label,
                                                        test_size=0.2,
                                                        random_state=0)
    #from sklearn.model_selection import cross_val_score
    #print cross_val_score(model, train, target, cv=10)

    model.fit(train_X,train_y)
    scores = model.predict(test_X)
    fp,tp,thresholds = metrics.roc_curve(test_y,scores,pos_label=1)
    ks = np.max(tp-fp)
    print "ROC K-S:{}".format(ks)
    print "AUC:{}".format(metrics.auc(fp, tp))
    return (model,ks)



def ROC2(model,train,target):
    from sklearn import metrics
    from sklearn.cross_validation import train_test_split

    ind_train = np.where(target > 0)[0]  # 获得训练数据为 1 的行

    label = target.iloc[ind_train]
    # print label

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
    train_X = pd.concat([train_X, train_X0])
    train_X = train_X.sort_index()

    test_X = pd.concat([test_X, test_X0])
    test_X = test_X.sort_index()

    train_y = pd.concat([train_y, train_y0])
    train_y = train_y.sort_index()

    test_y = pd.concat([test_y, test_y0])
    test_y = test_y.sort_index()

    model.fit(train_X, train_y)
    scores = model.predict(test_X)
    fp, tp, thresholds = metrics.roc_curve(test_y, scores, pos_label=1)
    ks = np.max(tp - fp)
    print "ROC2 K-S:{}".format(ks)
    print "AUC:{}".format(metrics.auc(fp, tp))
    return (model, ks)


def ROC3(model,train,target):
    from sklearn import metrics
    from sklearn.cross_validation import train_test_split

    ind_train = np.where(target > 0)[0]  # 获得训练数据为 1 的行

    label = target.iloc[ind_train]
    # print label

    trainX = train.iloc[ind_train]

    ind_train0 = np.where(target == 0)[0]  # 获得训练数据为 1 的行
    label0 = target.iloc[ind_train0]
    trainX0 = train.iloc[ind_train0]

    train_X, test_X, train_y, test_y = train_test_split(trainX,
                                                        label,
                                                        test_size=0.3,
                                                        random_state=0)

    train_X0, test_X0, train_y0, test_y0 = train_test_split(trainX0,
                                                            label0,
                                                            test_size=0.3,
                                                            random_state=0)
    train_X = pd.concat([train_X, train_X0])
    train_X = train_X.sort_index()

    test_X = pd.concat([test_X, test_X0])
    test_X = test_X.sort_index()

    train_y = pd.concat([train_y, train_y0])
    train_y = train_y.sort_index()

    test_y = pd.concat([test_y, test_y0])
    test_y = test_y.sort_index()

    model.fit(train_X, train_y)
    scores = model.predict(test_X)
    fp, tp, thresholds = metrics.roc_curve(test_y, scores, pos_label=1)
    ks = np.max(tp - fp)
    print "ROC3 K-S:{}".format(ks)
    print "AUC:{}".format(metrics.auc(fp, tp))
    return (model, ks)







