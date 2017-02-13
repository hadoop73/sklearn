# coding:utf-8

import pandas as pd
from GetData import getDatas2,getDatas3,getDatas,getData12
from WriteDatas import writeDatas

#  都是 pandas 的 DataFrame


import xgboost as xgb
#from xgboost import XGBClassifier as xgb
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


def evalerror(preds,d):
	labels = d.get_label()
	return 'KS',KS(pred=preds,y=labels)


def obj_fun(preds,dtrain):
	labels = dtrain.get_label()
	x = (preds - labels)
	grad = (1+2*labels)*x
	hess = (1+2*labels)
	return grad,hess

fair_constant = 0.7
def fair_obj(preds, dtrain):
    labels = dtrain.get_label()
    x = (preds - labels)
    den = abs(x) + fair_constant
    grad = fair_constant * x / (den)
    hess = fair_constant * fair_constant / (den * den)
    return grad, hess

#  0.33 train_data_5 train_data_bank_stage50
def XGBoost_(train=None,y=None,test=None,dtest_X=None,test_y=None,k=0,num_round=3500,
			 gamma=0.02,min_child_weight=1.1,max_depth=5,lamda=10,scale_pos_weight=3,
			 subsamp=0.7,col_bytree=0.7,col_bylevel=0.7,eta=0.01,file="aac"):

	param = {'booster':'gbtree',
			 'objective': 'binary:logistic',
			 #'eval_metric':'auc',
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
	with open('./test/a{}.txt'.format(file), 'a') as f:
		S = "gamma= " + str(gamma) + \
			" scale_pos_weight= " + str(scale_pos_weight) + \
			"  min_child_weight= " + str(min_child_weight) + \
			"  max_depth= " + str(max_depth) + \
			"  lamda= " + str(lamda) + \
			"\n" + \
			"subsamp= " + str(subsamp) + \
			"  col_bytree= " + str(col_bytree) + \
			"  col_bylevel= " + str(col_bylevel) + \
			"  eta= " + str(eta)
		f.writelines("{}\n".format(S))
	dtrain = xgb.DMatrix(train, label=y, missing=-9999)
	#cv_log = xgb.cv(param, dtrain,show_stdv=True,verbose_eval=1,feval=evalerror,num_boost_round=3500, nfold=5,early_stopping_rounds=10, seed=0)
	#num_round = 21#cv_log.shape[0]
	#cf = './featurescore/acvg{}.csv'.format(str(num_round))
	#cv_log.to_csv(cf)

	watchlist = [(dtrain, 'train'),(dtest_X,'eval')]
	#auc = cv_log['test-auc-mean'].max()
	bst = xgb.train(param, dtrain,num_round,watchlist,maximize=True,feval=evalerror,early_stopping_rounds=50)
	# make prediction
	dtest = xgb.DMatrix(test,missing=-9999)
	preds = bst.predict(dtest,ntree_limit=bst.best_ntree_limit)
	p = bst.predict(dtrain, ntree_limit=bst.best_ntree_limit)


	scores = bst.predict(dtest_X,ntree_limit=bst.best_ntree_limit)
	fp, tp, thresholds = metrics.roc_curve(test_y, scores, pos_label=1)
	auc = metrics.auc(fp, tp)
	ks = KS(y=test_y.label,pred=scores)
	kk = int(ks * 10000000000) % 10000
	print "K-S:{}".format(ks)
	print "AUC:{}".format(auc)

	with open('./test/a{}.txt'.format(file), 'a') as f:
		S =  "  best_ntree_limit:" + str(bst.best_ntree_limit) + \
			 "   best_iteration= "+str(bst.best_iteration)+ \
			"\nfeatures scores: " + str(kk)
		f.writelines("{}\n".format(S))
		f.writelines("K-S:{}\n".format(ks))
		f.writelines("AUC:{}\n\n".format(metrics.auc(fp, tp)))

	res = writeDatas(preds, test, "xgk_{}".format(str(kk)))

	res.columns = ['label'+str(kk)]
	y['label'+str(kk)] = p
	y = pd.concat([y,res])
	y.drop('label',axis=1,inplace=True)
	y = y.reset_index()
	try:
		ypred = pd.read_csv("./test/y/a{}.csv".format(file))
		y = pd.merge(y,ypred,on='userid')
	except:
		pass
	finally:
		y.to_csv("./test/y/a{}.csv".format(file),index=None)

	# get feature score
	feature_score = bst.get_fscore()
	feature_score = sorted(feature_score.items(), key=lambda x: x[1], reverse=True)
	fs = []
	for (key, value) in feature_score:
		fs.append("{0},{1}\n".format(key, value))
	print "features scores:",kk
	ff = './test/feature_score_{0}.csv'.format(kk)
	with open(ff, 'w') as f:
		f.writelines("feature,score\n")
		f.writelines(fs)


#  0.33 train_data_5 train_data_bank_stage50
def XGBoost_gbdm(train=None,y=None,test=None,dtest_X=None,test_y=None,k=0,num_round=3500,
			 gamma=0.02,min_child_weight=1.1,max_depth=5,lamda=10,scale_pos_weight=3,
			 subsamp=0.7,col_bytree=0.7,col_bylevel=0.7,eta=0.01,file="aac"):

	param = {'booster':'gbtree',
			 'objective': 'binary:logistic',
			 #'eval_metric':'auc',
			 'scale_pos_weight':scale_pos_weight,
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
	with open('./findx/af{}.txt'.format(file), 'a') as f:
		S = "gamma= " + str(gamma) + \
			"  scale_pos_weight= " + str(scale_pos_weight) + \
			"  min_child_weight= " + str(min_child_weight) + \
			"  max_depth= " + str(max_depth) + \
			"  lamda= " + str(lamda) + \
			"\n" + \
			"subsamp= " + str(subsamp) + \
			"  col_bytree= " + str(col_bytree) + \
			"  col_bylevel= " + str(col_bylevel) + \
			"  eta= " + str(eta)
		f.writelines("{}\n".format(S))
	dtrain = xgb.DMatrix(train, label=y.label, missing=-9999)
	#cv_log = xgb.cv(param, dtrain,show_stdv=True,verbose_eval=1,feval=evalerror,num_boost_round=3500, nfold=5,early_stopping_rounds=10, seed=0)
	#num_round = 21#cv_log.shape[0]
	#cf = './featurescore/acvg{}.csv'.format(str(num_round))
	#cv_log.to_csv(cf)

	watchlist = [(dtrain, 'train'),(dtest_X,'eval')]
	bst = xgb.train(param, dtrain,num_round,watchlist,maximize=True,feval=evalerror,early_stopping_rounds=50)

	scores = bst.predict(dtest_X,ntree_limit=bst.best_ntree_limit)
	fp, tp, thresholds = metrics.roc_curve(test_y, scores, pos_label=1)
	auc = metrics.auc(fp, tp)
	ks = KS(y=test_y.label,pred=scores)
	kk = int(ks * 10000000000) % 10000
	print "K-S:{}".format(ks)
	print "AUC:{}".format(auc)

	with open('./findx/af{}.txt'.format(file), 'a') as f:
		S =  "  best_ntree_limit:" + str(bst.best_ntree_limit) + \
			 "   best_iteration= "+str(bst.best_iteration)+ \
			"\nfeatures scores: " + str(kk)
		f.writelines("{}\n".format(S))
		f.writelines("K-S:{}\n".format(ks))
		f.writelines("AUC:{}\n\n".format(metrics.auc(fp, tp)))
		#f.writelines("AUC:{}\n\n".format(metrics.auc(fp, tp)))

	return ks,auc,bst.best_ntree_limit


"""
"""


def test_xgb(f,
		gamma,
	    scale_pos_weight,
		min_child_weight,
		max_depth,
		lamda,
		subsamp,
		col_bytree,
		col_bylevel,
		eta,num_round,func=None):

	for fd in [f]:
		train, target, test = getDatas(fd)
		train_X, train_y, test_X, test_y, test = func(fd)
		dtest_X = xgb.DMatrix(test_X, label=test_y, missing=-9999)
		del train_X, train_y
		with open('./test/a{}.txt'.format(fd), 'a') as f:
			f.writelines("data: " + fd + "\n")
		# kk = XGBoost_(dtrain=dtrain,test=test)
		kk = XGBoost_(train=train,y=target, test=test, test_y=test_y, dtest_X=dtest_X,
						  gamma=gamma,
					  	  scale_pos_weight=scale_pos_weight,
						  min_child_weight=min_child_weight,
						  max_depth=max_depth,
						  lamda=lamda,
						  subsamp=subsamp,
						  col_bytree=col_bytree,
						  col_bylevel=col_bylevel,
						  eta=eta, num_round=num_round,
					  	  file=fd
					  )
		# part_xg(kk,fd)
		del train, target, test,test_X, test_y

def train_xgb():
	dirs = ['', 'data_browser_dg_decode2n5', 'train_dg_bank2', 'train_12_bank2', 'train_12_brow2', 'train_data_12',
			'train_data_new_kp_e2', 'train_data_new', 'train_data_new_kp_e', 'train_data_new_knnpca_dummy']
	i = 1
	"""
	gamma= 0.2  min_child_weight= 1.5  max_depth= 4  lamda= 260
subsamp= 0.8  col_bytree= 0.5  col_bylevel= 0.8  eta= 0.12
  best_ntree_limit:186   best_iteration= 185
	"""
	for fd in ['train_4']:
		#train, target, test = getDatas(fd)
		train_X, train_y, test_X, test_y, test = getData12(fd)
		dtest_X = xgb.DMatrix(test_X, label=test_y, missing=-9999)


		with open('./featurescore/a.txt', 'a') as f:
			f.writelines("data: " + fd + " 部分数据：" + str(i) + "\n")
		# kk = XGBoost_(dtrain=dtrain,test=test)
		kk = XGBoost_gbdm(train=train_X, y=train_y,test=test, test_y=test_y, dtest_X=dtest_X,
						  gamma=0.2,
						  min_child_weight=1.5,
						  max_depth=4,
						  lamda=260,
						  subsamp=0.8,
						  col_bytree=0.5,
						  col_bylevel=0.8,
						  eta=0.12,num_round=186
						  )
		# part_xg(kk,fd)
		i += 1

def find_X(getDataFun=None):
	gamma = [0.08,0.15,0.1,0.2] # default 0
	eta = [0.01,0.05,0.1,0.12,0.18] # 0.01-0.2
	max_depth = np.arange(4, 11)  # 3-10

	subsamp = [0.9,0.7,0.8,0.6] # 0.5-1
	col_bytree = [0.5,0.7,0.8,0.6] # 0.5-1
	col_bylevel = [0.5,0.7,0.8,0.6] # default 1 colsample_bylevel exceed bound [0,1]
	min_child_weight = [0.5,0.8,1.0,1.2,1.5] # default 1
	lmda = np.arange(10, 400,50)
	scale_pos_weight = [3,4]

	import random
	random.shuffle(gamma)
	random.shuffle(eta)
	random.shuffle(max_depth)
	random.shuffle(subsamp)
	random.shuffle(col_bytree)
	random.shuffle(col_bylevel)
	random.shuffle(min_child_weight)
	random.shuffle(lmda)


	"""
	XGBoost_gbdm(dtrain=None,test=None,dtest_X=None,test_y=None,k=0,num_round=3500,
			 gamma=0.02,min_child_weight=1.1,max_depth=5,lamda=10,
			 subsamp=0.7,col_bytree=0.7,col_bylevel=0.7,eta=0.01):
	"""

	for fd in ["data_bill_bank_cut10_allSB_selectCD_loan_time_f43","data_all_bill_ban_cut10_allSB_selectCD_loan_time"]:
			# train, target, test = getDatas(fd)
			with open('./findx/af{}.txt'.format(fd), 'a') as f:
				f.writelines("data: " + fd + "\n")
			# kk = XGBoost_(dtrain=dtrain,test=test)
			for i in range(1,11):
				with open('./findx/af{}.txt'.format(fd), 'a') as f:
					f.writelines("times: " + str(i) + "\n")
				train_X, train_y, test_X, test_y, test = getDataFun(fd)
				dtest_X = xgb.DMatrix(test_X, label=test_y, missing=-9999)
				ks,auc,num_round = XGBoost_gbdm(train=train_X,y=train_y, test=test, test_y=test_y, dtest_X=dtest_X,
								  gamma=gamma[i%len(gamma)],
								  scale_pos_weight=scale_pos_weight[i % len(scale_pos_weight)],
								  min_child_weight=min_child_weight[i % len(min_child_weight)],
								  max_depth=max_depth[i % len(max_depth)],
								  lamda = lmda[i % len(lmda)],
								  subsamp=subsamp[i % len(subsamp)],
								  col_bytree = col_bytree[i % len(col_bytree)],
								  col_bylevel = col_bylevel[i % len(col_bylevel)],
								  eta = eta[i % len(eta)],file=fd,
								  )
				del train_X, train_y, test_X, test_y, test,dtest_X
				if (ks > 0.46) & (auc >0.79):
					test_xgb(fd,
							gamma=gamma[i % len(gamma)],
							scale_pos_weight=scale_pos_weight[i % len(scale_pos_weight)],
							min_child_weight=min_child_weight[i % len(min_child_weight)],
							max_depth=max_depth[i % len(max_depth)],
							lamda=lmda[i % len(lmda)],
							subsamp=subsamp[i % len(subsamp)],
							col_bytree=col_bytree[i % len(col_bytree)],
							col_bylevel=col_bylevel[i % len(col_bylevel)],
							eta=eta[i % len(eta)],
							num_round=num_round,func=getDataFun,
					)
				# part_xg(kk,fd)
				if i%8==0:
					random.shuffle(gamma)
					random.shuffle(eta)
					random.shuffle(max_depth)
					random.shuffle(subsamp)
					random.shuffle(col_bytree)
					random.shuffle(col_bylevel)
					random.shuffle(min_child_weight)
					random.shuffle(lmda)

def getD():
	b = pd.read_csv("../data/train.csv")
	d = b[['userid','browse_count_time_lt','pre_amount_minus_min',
		   'stg4_browser_data_min','browse_count_min','browse_count_median',
		   'time_mean','time_median','un0','stg4_examount0_median','stg5_examount0_median',
		   'stg5_examount0_mean','stg4_examount0_mean']]
	print d.head()
	print d.shape
	d.to_csv('../data/train/new.csv',index=None)

def concate_data():
	a = pd.read_csv("../data/train/traina.csv")
	b = pd.read_csv("../data/train/testa.csv")
	d = pd.concat([a,b],axis=0)
	d['userid'] = d['userid'].astype('int')
	d.replace(-999,-9999,inplace=True)
	print d.head()
	print d.shape
	d.to_csv("../data/train/bill_42.csv",index=None)

def merge_2():

	names_loan_time = ['userid', 'loan_time']
	loan_time_train = pd.read_csv("../../pcredit/train/loan_time_train.txt", header=None)
	loan_time_test = pd.read_csv("../../pcredit/test/loan_time_test.txt", header=None)
	loan_time = pd.concat([loan_time_train, loan_time_test], axis=0)
	loan_time.columns = names_loan_time

	#m = pd.read_csv('../data/man/allSB0.csv')
	f = pd.read_csv("../data/data_all_bill_ban_cut10_allSB_selectCD.csv")

	c = pd.merge(f,loan_time,on='userid',how='outer')
	c.fillna(-9999, inplace=True)
	print c.head()
	print c.shape
	c.to_csv("../data/data_all_bill_ban_cut10_allSB_selectCD_loan_time.csv",index=None)

if __name__=='__main__':
	for func in [getData12,getDatas2,getDatas3]:
		find_X(func)
	#merge_2()


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



