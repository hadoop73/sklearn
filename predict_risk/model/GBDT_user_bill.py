# coding:utf-8


from GetData import getDatas
from WriteDatas import writeDatas

#  都是 pandas 的 DataFrame

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
font = FontProperties(fname='/usr/share/fonts/truetype/arphic/ukai.ttc',size=14)

#  都是 pandas 的 DataFrame

from GetUserBillData import getUserBillData

train,target,test =  getUserBillData()

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor


clf = GradientBoostingRegressor(n_estimators=20, learning_rate=0.1,
            max_depth=3, random_state=0, loss='ls')  # .fit(train, target)

from ROC import ROC

clf = ROC(clf,train,target)

result = clf.predict(test)

#print result

#writeDatas(result,test,"00")








