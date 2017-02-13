# coding:utf-8



"""
银行流水记录bank_detail.txt。
共5个字段，其中，第2个字段，时间戳为0表示时间未知；
第3个字段，交易类型有两个值，1表示支出、0表示收入；
第5个字段，工资收入标记为1时，表示工资收入。

用户id,时间戳,交易类型,交易金额,工资收入标记
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
userid	time	extype	examount	mark
6965	5894316387	0	13.756664	0
6965	5894321388	1	13.756664	0
6965	5897553564	0	14.449810	0
6965	5897563463	1	10.527763	0
6965	5897564598	1	13.651303	0
"""

"""
银行卡,分时间段统计支出的次数，支出的平均金额，
统计是否有稳定收入，也就是统计收入的次数
统计支出和收入的时间差，表示缺钱程度
统计收入与 bill 数据中本期账单的差值
"""



def getBankData():
    names = ['userid', 'time', 'extype', 'examount', 'mark']
    bank_detail_train = pd.read_csv("../../pcredit/train/bank_detail_train.txt", header=None)
    bank_detail_test = pd.read_csv("../../pcredit/test/bank_detail_test.txt", header=None)

    bank_detail = pd.concat([bank_detail_train, bank_detail_test])
    bank_detail.columns = names
    del bank_detail_train, bank_detail_test
    return bank_detail

def drop_bank_duplicates(bank_data):
    print bank_data.shape
    bank_data.drop_duplicates(['userid','time','extype','examount'],inplace=True)
    print bank_data.head()
    print bank_data.shape
    bank_data.to_csv('data/train/bank_diff.csv',index=None)

bank_data = pd.read_csv('data/train/bank_diff.csv')
users = list(bank_data.userid.unique())

import numpy as np
import pandas as pd
from pandas import DataFrame,Series
import seaborn as sns
import matplotlib.pyplot as plt

#sns.kdeplot(bank_data['examount'][bank_data.extype==1],color='r')
#plt.title('examount 1')
#plt.show()


t = bank_data['examount'][bank_data.extype==0].describe(percentiles=[ 0.2,  0.4,  0.6,  0.8])
split_point_col = {}
split_point_col[0] = [0, int(t['20%']),  int(t['40%']),
                        int(t['60%']),  int(t['80%']),  1e11]

t = bank_data['examount'][bank_data.extype==1].describe(percentiles=[ 0.2,  0.4,  0.6,  0.8])
split_point_col[1] = [0, int(t['20%']),  int(t['40%']),
                        int(t['60%']),  int(t['80%']),  1e11]

stage = ['stg'+str(i)+"_" for i in range(1,6)]

def cut_10(u):
    d = {'userid':u}
    for c in [0,1]:
        dt = bank_data[(bank_data.userid == u) & (bank_data.extype == c)]
        for i in range(0,5):
            stg = stage[i]
            di = dt[(split_point_col[c][i] <= dt.examount) & (dt.examount < split_point_col[c][i + 1])]
            d[stg + "examount_" + str(c) + '_cnt'] = di['examount'].count()
            d[stg + "examount_" + str(c) + '_min'] = di['examount'].min()
            d[stg + "examount_" + str(c) + '_max'] = di['examount'].max()
            d[stg + "examount_" + str(c) + '_mean'] = di['examount'].mean()

    for i in range(0, 5):
        stg = stage[i]
        d[stg + "examount_" + 'a' + '_cnt'] = d[stg + "examount_" + '0' + '_cnt'] - d[stg + "examount_" + '1' + '_cnt']
        d[stg + "examount_" + 'a' + '_min'] = d[stg + "examount_" + '0' + '_min'] - d[stg + "examount_" + '1' + '_min']
        d[stg + "examount_" + 'a' + '_max'] = d[stg + "examount_" + '0' + '_max'] - d[stg + "examount_" + '1' + '_max']
        d[stg + "examount_" + 'a' + '_mean'] = d[stg + "examount_" + '0' + '_mean'] - d[stg + "examount_" + '1' + '_mean']

    print d
    return d

def multi():
    from multiprocessing import Pool
    pool = Pool(8)
    u = users[:100]
    #  获得 split time 的数据统计信息
    rst = pool.map(cut_10,users)
    pool.close()
    pool.join()
    features = pd.DataFrame(rst)
    fileName = 'data/train/bank_cut10.csv'
    #  获得
    print features.head()
    print "\t", fileName
    print features.shape
    features.to_csv(fileName,index=None)

if __name__=='__main__':
    pass
    #data = getBankData()
    #drop_bank_duplicates(data)
    multi()
    #cut_10(3456)











