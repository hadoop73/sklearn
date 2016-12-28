# coding:utf-8

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


from GetData import getDatas
from WriteDatas import writeDatas

#  都是 pandas 的 DataFrame

train,target,test =  getDatas('bill_browser_user_data')

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor

from ROC import ROC

def adTree():
    clf = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                              n_estimators=300)
    clf = ROC(clf, train, target)
    result = clf.predict(test)


rf = RandomForestRegressor(max_depth=4, random_state=2,n_estimators=100)

rf = ROC(rf,train,target)



# 输出测试集用户逾期还款概率，predict_proba会输出两个概率，取‘1’的概率


#print result

#writeDatas(result,test,"700")








