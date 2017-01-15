# coding:utf-8


from GetData import getXGBoostDatas,getDatas
from WriteDatas import writeDatas

#  都是 pandas 的 DataFrame

train,target,test =  getDatas("train_data_")

import xgboost as xgb
import numpy as np



from sklearn import metrics
from sklearn.cross_validation import train_test_split

train_X, test_X, train_y, test_y = train_test_split(train,
                                                    target.label,
                                                    test_size=0.2,
                                                    random_state=0)

#  转换数据
#train.drop(['userid'],axis=1,inplace=True)
dtrain = xgb.DMatrix(train_X,label=train_y)

#test.drop(['userid'],axis=1,inplace=True)
dtest = xgb.DMatrix(test)


#  参数设置


param={'booster':'gbtree',
	    'objective': 'binary:logistic',
	    'eval_metric':'auc',
	    'gamma':0.1,
	    'min_child_weight':1.1,
	    'max_depth':5,
	    'lambda':10,
	    'subsample':0.7,
	    'colsample_bytree':0.7,
	    'colsample_bylevel':0.7,
	    'eta': 0.01,
	    'tree_method':'exact',
	    'seed':0,
	    'nthread':12
	    }

num_round = 1500
watchlist = [(dtrain,'train')]
bst = xgb.train(param, dtrain, num_round,evals=watchlist)
# make prediction
preds = bst.predict(dtest)

dtest_X = xgb.DMatrix(test_X)
scores = bst.predict(dtest_X)

fp, tp, thresholds = metrics.roc_curve(test_y, scores, pos_label=1)
ks = np.max(tp-fp)
print "K-S:{}".format(ks)
print "AUC:{}".format(metrics.auc(fp, tp))


#  写入文件
writeDatas(preds,test,"xg{}".format(ks))


# get feature score
feature_score = bst.get_fscore()
feature_score = sorted(feature_score.items(), key=lambda x: x[1], reverse=True)
fs = []

allFeatures = []
for (key, value) in feature_score:
	fs.append("{0},{1}\n".format(key, value))
	allFeatures.append(key)

with open('./featurescore/feature_score_{0}.csv'.format('x4'), 'w') as f:
	f.writelines("feature,score\n")
	f.writelines(fs)


#  删除排名靠后的 5 10 15 个特征

select_n = [-5,-10,-15]

from GBDT import gbdt

#  不删除数据的结果
features = allFeatures
print "features size:",len(features)
train_data = train[features]
test_data = test[features]
gbdt(train=train_data, target=target, test=test_data, n=0)

for n in select_n:
	features = allFeatures[:n]
	train_data = train[features]
	test_data = test[features]
	gbdt(train=train_data,target=target,test=test_data,n=n)


"""
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

"""



