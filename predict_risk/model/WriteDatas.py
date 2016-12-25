# coding:utf-8

import numpy as np
import pandas as pd

def  writeDatas(data,test,name):
    result = pd.DataFrame(data)
    result.index = test.index
    result.columns = ['probability']
    print result.head(5)
    # 输出结果
    result.to_csv('../data/result_{}.csv'.format(name))








