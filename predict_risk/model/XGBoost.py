# coding:utf-8


from GetData import getXGBoostDatas,getDatas,getDatas2
from WriteDatas import writeDatas

#  都是 pandas 的 DataFrame


import xgboost as xgb
import numpy as np
from sklearn import metrics


from KS import KS


'''
gamma=0.1,min_child_weight=1.1,max_depth=5,lamda=10,
			 subsamp=0.7,col_bytree=0.7,col_bylevel=0.7,eta=0.01
	param = {'booster': 'gbtree',
			 'objective': 'binary:logistic',
			 'eval_metric': 'auc',
			 'gamma': gamma,
			 'min_child_weight': min_child_weight,
			 'max_depth': max_depth,
			 'lambda': lamda,
			 'subsample': subsamp,
			 'colsample_bytree': col_bytree,
			 'colsample_bylevel': col_bylevel,
			 'eta': eta,
			 'tree_method': 'exact',
			 'seed': 0,
			 'nthread': 12
			 }
	num_round = 1000
'''


def XGBoost_(dtrain=None,test=None,dtest_X=None,test_y=None,k=0,
			 gamma=0.1,min_child_weight=1.1,max_depth=5,lamda=10,
			 subsamp=0.7,col_bytree=0.7,col_bylevel=0.7,eta=0.01):

	param = {'booster': 'gbtree',
			 'objective': 'binary:logistic',
			 'eval_metric': 'auc',
			 'gamma': gamma,
			 'min_child_weight': min_child_weight,
			 'max_depth': max_depth,
			 'lambda': lamda,
			 'subsample': subsamp,
			 'colsample_bytree': col_bytree,
			 'colsample_bylevel': col_bylevel,
			 'eta': eta,
			 'tree_method': 'exact',
			 'seed': 0,
			 'nthread': 12
			 }
	num_round = 3500
	watchlist = [(dtrain, 'train')]
	bst = xgb.train(param, dtrain, num_round, evals=watchlist,early_stopping_rounds=50)
	# make prediction
	dtest = xgb.DMatrix(test)
	preds = bst.predict(dtest)


	scores = bst.predict(dtest_X)

	fp, tp, thresholds = metrics.roc_curve(test_y, scores, pos_label=1)
	ks = KS(y=test_y,score=scores)
	print "K-S:{}".format(ks)
	print "AUC:{}".format(metrics.auc(fp, tp))

	with open('./featurescore/a.txt', 'a') as f:
		S = "gamma= "+str(gamma)+\
			"  min_child_weight= "+str(min_child_weight)+\
			"  max_depth= "+str(max_depth)+\
			"  lamda= "+str(lamda)+\
			"  subsamp= "+str(subsamp)+\
			"  col_bytree= "+str(col_bytree)+\
			"  col_bylevel= "+str(col_bylevel)+\
			"  eta= "+str(eta)
		f.writelines("{}\n".format(S))
		f.writelines("K-S:{}\n".format(ks))
		f.writelines("AUC:{}\n\n".format(metrics.auc(fp, tp)))

	#  写入文件
	writeDatas(preds, test, "xgk{}".format(str(ks)))

	# get feature score
	feature_score = bst.get_fscore()
	feature_score = sorted(feature_score.items(), key=lambda x: x[1], reverse=True)
	fs = []

	for (key, value) in feature_score:
		fs.append("{0},{1}\n".format(key, value))

	kk = int(ks*10000000000)%1000
	print "features scores:",kk
	ff = './featurescore/feature_score_{0}.csv'.format(kk)
	with open(ff, 'w') as f:
		f.writelines("feature,score\n")
		f.writelines(fs)
	return ff

#  0.33 train_data_5 train_data_bank_stage50

from XGBoost_Part import part

if __name__=='__main__':
	dirs = ['train_data_new_kp_e2','train_data_new','train_data_new_kp_e','train_data_new_knnpca_dummy']
	for fd in ['train_data_12']:
		for kf in [0.2]:
			train_X, test_X, train_y, test_y, test = getDatas2(dir=fd,k=kf)

			print "train data :",train_X.shape,"  data: ",fd

			dtrain = xgb.DMatrix(train_X,label=train_y)
			dtest_x = xgb.DMatrix(test_X)
			dtest =  xgb.DMatrix(test_X)
			with open('./featurescore/a.txt', 'a') as f:
				f.writelines("data: "+fd+ "  数据比例："+str(kf)+"\n")
			kk = XGBoost_(dtrain=dtrain,test=test,dtest_X=dtest_x,test_y=test_y)
			del train_X, test_X, train_y, test_y, test,dtrain,dtest_x,dtest
			try:
				print "part"
				part(fd,kk)
			except:
				pass




def find_X():
	gamma = np.linspace(0.1, 0.7, 6)
	eta = np.linspace(0.01, 0.2, 10)
	max_depth = np.arange(3, 9)

	subsamp = np.linspace(0.5, 1, 5)
	col_bytree = np.linspace(0.5, 1, 5)
	col_bylevel = np.linspace(0.5, 1,5)
	min_child_weight = np.linspace(0.8, 2, 5)
	lmda = np.arange(5, 19,2)

	import random
	random.shuffle(gamma)
	random.shuffle(eta)
	random.shuffle(max_depth)
	random.shuffle(subsamp)
	random.shuffle(col_bytree)
	random.shuffle(col_bylevel)
	random.shuffle(min_child_weight)
	random.shuffle(lmda)

	from multiprocessing import Pool

	rst = []
	pool = Pool(12)
	for i in range(36):
		pool.apply_async(XGBoost_,args=(gamma[i%len(gamma)],
										min_child_weight[i%len(min_child_weight)],
										max_depth[i % len(max_depth)],
										lmda[i % len(lmda)],
										subsamp[i % len(subsamp)],
										col_bytree[i % len(col_bytree)],
										col_bylevel[i % len(col_bylevel)],
										eta[i % len(eta)],
										))
	pool.close()
	pool.join()


"""
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

"""







#  删除排名靠后的 5 10 15 个特征

#select_n = [-5,-10,-15]

"""
for n in select_n:
	features = allFeatures[:n]
	train_data = train[features]
	test_data = test[features]
	gbdt(train=train_data,target=target,test=test_data,n=n)

"""

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



