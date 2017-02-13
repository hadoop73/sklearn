# coding:utf-8



import xgboost as xgb
import numpy as np
import pandas as pd


from WriteDatas import writeDatas

from KS import KS
import xgboost as xgb
import numpy as np
from sklearn import metrics
#  都是 pandas 的 DataFrame

def XGBoost_part(dtrain=None,test=None,dtest_X=None,test_y=None,k=0,
			 gamma=0.02,min_child_weight=1.1,max_depth=5,lamda=100,
			 subsamp=0.7,col_bytree=0.7,col_bylevel=0.7,eta=0.01):

	param = {'booster':'gbtree',
			 'objective': 'binary:logistic',
			 'eval_metric':'auc',
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
	cv_log = xgb.cv(param, dtrain, num_boost_round=3500, nfold=5,early_stopping_rounds=50, seed=0)
	num_round = cv_log.shape[0]
	cf = './featurescore/cvg{}.csv'.format(str(num_round))
	cv_log.to_csv(cf)
	watchlist = [(dtrain, 'train')]
	#auc = cv_log['test-auc-mean'].max()
	bst = xgb.train(param, dtrain, num_round,evals=watchlist,early_stopping_rounds=50)
	# make prediction
	dtest = xgb.DMatrix(test,missing=-9999)
	preds = bst.predict(dtest)

	scores = bst.predict(dtrain,ntree_limit=bst.best_ntree_limit)
	fp, tp, thresholds = metrics.roc_curve(test_y, scores, pos_label=1)
	ks = KS(y=test_y,score=scores)
	kk = int(ks * 10000000000) % 1000
	print "K-S:{}".format(ks)
	print "AUC:{}".format(metrics.auc(fp, tp))

	with open('./featurescore/a.txt', 'a') as f:
		S = "gamma= "+str(gamma)+\
			"  min_child_weight= "+str(min_child_weight)+\
			"  max_depth= "+str(max_depth)+\
			"  lamda= "+str(lamda)+\
			"\n" + \
			"subsamp= "+str(subsamp)+\
			"  col_bytree= "+str(col_bytree)+\
			"  col_bylevel= "+str(col_bylevel)+\
			"  eta= "+str(eta) + \
			"  ntree= "+str(bst.best_ntree_limit)+ \
			"\nfeatures scores: " + str(kk)
		f.writelines("{}\n".format(S))
		f.writelines("K-S:{}\n".format(ks))
		f.writelines("AUC:{}\n\n".format(metrics.auc(fp, tp)))
		#f.writelines("AUC:{}\n\n".format(metrics.auc(fp, tp)))
	#  写入文件
	writeDatas(preds, test, "xgk{}".format(str(ks)))

	# get feature score
	feature_score = bst.get_fscore()
	feature_score = sorted(feature_score.items(), key=lambda x: x[1], reverse=True)
	fs = []

	for (key, value) in feature_score:
		fs.append("{0},{1}\n".format(key, value))

	print "features scores:",kk
	ff = './featurescore/feature_score_{0}.csv'.format(kk)
	with open(ff, 'w') as f:
		f.writelines("feature,score\n")
		f.writelines(fs)
	return kk


from GetData import getDatas

def part(dir,fdir):
    features = pd.read_csv(fdir)
    for i in [10,20]:
		flist = list(features[features.score>=i]['feature'])
		train, target, test = getDatas(dir=dir)
		train_part = train[flist]
		test_part = test[flist]
		dtrain = xgb.DMatrix(train_part, label=target, missing=-9999)
		print "train_part size: ",train_part.shape
		print "test_part size: ",test_part.shape
		with open('./featurescore/a.txt', 'a') as f:
			f.writelines("数据： {} 特征路径：{}  特征分:{}\n".format(dir,fdir,i))
		XGBoost_part(dtrain=dtrain, test=test_part, test_y=target)


def part_xg(f,d):
	ff = './featurescore/feature_score_{0}.csv'.format(f)
	part(dir=d, fdir=ff)


if __name__=='__main__':
	for f,d in [(930,'all_data3')]:
		ff = './featurescore/feature_score_{0}.csv'.format(f)
		part(dir=d,fdir=ff)


