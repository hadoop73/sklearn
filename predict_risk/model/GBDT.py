# coding:utf-8


from GetData import getDatas
from WriteDatas import writeDatas

#  都是 pandas 的 DataFrame

train,target,test = getDatas('bill_browser_user_data')

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor


clf = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
            max_depth=4, random_state=0, loss='ls')  # .fit(train, target)

from ROC import ROC

clf = ROC(clf,train,target)

result = clf.predict(test)

#print result

writeDatas(result,test,"700")








