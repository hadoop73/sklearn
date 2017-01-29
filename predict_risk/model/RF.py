# coding:utf-8


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier



train,target,test =  getDatas("train_data_")

from sklearn import metrics
from sklearn.cross_validation import train_test_split

train_X, test_X, train_y, test_y = train_test_split(train,
                                                    target.label,
                                                    test_size=0.2,
                                                    random_state=0)


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=1)


rf.fit(train_X, train_y)

score = rf.predict_proba(test_X)[:,1]


ans = rf.predict_proba(test)[:,1]