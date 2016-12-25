# coding:utf-8


from GetData import getDatas
from WriteDatas import writeDatas

#  都是 pandas 的 DataFrame

train,target,test = getDatas()

import xgboost as xgb


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
writeDatas(preds,test,"xg")














