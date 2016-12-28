# coding:utf-8


from GetData import getDatas
from WriteDatas import writeDatas

#  都是 pandas 的 DataFrame

train,target,test =  getDatas('bill_browser_user_data')

import xgboost as xgb
import numpy as np

#  转换数据
dtrain = xgb.DMatrix(train.values,label=target.values)
dtest = xgb.DMatrix(test.values)


#  参数设置
param = {'bst:max_depth':2, 'bst:eta':1, 'silent':1, 'objective':'binary:logistic' }
param['nthread'] = 4
num_round = 4
bst = xgb.train(param, dtrain, num_round)
# make prediction
preds = bst.predict(dtest)

#  写入文件
writeDatas(preds,test,"0")


from sklearn import metrics
from sklearn.cross_validation import train_test_split

train_X, test_X, train_y, test_y = train_test_split(train.values,
                                                    target.values,
                                                    test_size=0.2,
                                                    random_state=0)

#  转换数据
dtrain_X = xgb.DMatrix(train_X,label=train_y)
dtest_X = xgb.DMatrix(test_X)
bst = xgb.train(param, dtrain_X, num_round)
# make prediction
scores = bst.predict(dtest_X)
fp, tp, thresholds = metrics.roc_curve(test_y, scores, pos_label=1)
print "K-S:{}".format(np.max(tp-fp))
print "AUC:{}".format(metrics.auc(fp, tp))





