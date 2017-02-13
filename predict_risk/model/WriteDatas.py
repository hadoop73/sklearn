# coding:utf-8

import numpy as np
import pandas as pd

def  writeDatas(data,test,name='ok'):
    result = pd.DataFrame(data)
    result.index = test.index.astype(int)
    result.columns = ['probability']
    result['probability'] = result['probability'].apply(lambda x:np.abs(x))
    print "Datas writed:"
    print result.head(5)
    # 输出结果
    result.to_csv('../data/result_{}.csv'.format(name))
    return result








