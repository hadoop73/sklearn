# coding:utf-8


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier



from sklearn import metrics
from sklearn.cross_validation import train_test_split

from GetData import getXGBoostDatas,getDatas,getDatas2
from WriteDatas import writeDatas
from KS import KS

train_X, test_X, train_y, test_y, test = getDatas2(dir='train_data_12')


from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor

for c in ['entropy','gini']:
    for md in [6,8,10]:
        for n in [300,600,800,1000,1200]:

            try:
                rf = RandomForestClassifier(n_estimators=n,criterion=c,warm_start=True,max_depth=md,max_features=0.6,min_samples_leaf=5,n_jobs=12,random_state=0)


                rf.fit(train_X, train_y)

                score = rf.predict_proba(test_X)[:,1]

                fp, tp, thresholds = metrics.roc_curve(test_y.values, score, pos_label=1)
                ks = KS(y=test_y, score=score)
                print "K-S:{}".format(ks)
                print "AUC:{}".format(metrics.auc(fp, tp))

                ans = rf.predict_proba(test)[:,1]

                with open('./featurescore/a.txt', 'a') as f:
                    S = "criterion= " + str(c) + \
                        "  n_estimators= " + str(n) + \
                        "  max_depth= " + str(md)
                    f.writelines("{}\n".format(S))
                    f.writelines("K-S:{}\n".format(ks))
                    f.writelines("AUC:{}\n\n".format(metrics.auc(fp, tp)))

                writeDatas(ans, test, "rf{}".format(str(ks)))
            except:
                S = "criterion= " + str(c) + \
                    "  n_estimators= " + str(n) + \
                    "  max_depth= " + str(md)
                print "Eorr",S
                pass








