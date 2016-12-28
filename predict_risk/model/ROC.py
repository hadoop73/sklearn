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
    train_X, test_X, train_y, test_y = train_test_split(train,
                                                        target,
                                                        test_size=0.2,
                                                        random_state=0)
    #from sklearn.model_selection import cross_val_score
    #print cross_val_score(model, train, target, cv=10)
    model.fit(train_X,train_y)
    scores = model.predict(test_X)
    fp,tp,thresholds = metrics.roc_curve(test_y,scores,pos_label=1)
    print "K-S:{}".format(np.max(tp-fp))
    print "AUC:{}".format(metrics.auc(fp, tp))
    return model















