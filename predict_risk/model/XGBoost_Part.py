# coding:utf-8



import xgboost as xgb
import numpy as np
import pandas as pd


from GetData import getXGBoostDatas,getDatas
from WriteDatas import writeDatas

#  都是 pandas 的 DataFrame

train,target,test =  getDatas("train_data_")

features = pd.read_csv('./featurescore/feature_score_{0}.csv'.format(0))

print features.head()

from predict_risk.model.XGBoost import XGBoost_

for i in [10,30,50,80,100]:

    f =  list(features[features.score>i]['feature'])
    train_part = train[f]
    test_part = test[f]

    print "train_part size: ",train_part.shape
    from sklearn import metrics
    from sklearn.cross_validation import train_test_split

    train_X, test_X, train_y, test_y = train_test_split(train_part,
                                                        target.label,
                                                        test_size=0.2,
                                                        random_state=0)
    dtrain = xgb.DMatrix(train_X, label=train_y)

    dtest_X = xgb.DMatrix(test_X)

    XGBoost_(dtrain=dtrain,test=test_part,dtest_X=dtest_X,test_y=test_y,k=i)










