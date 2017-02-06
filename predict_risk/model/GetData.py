# coding:utf-8

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report




def KK(x):
    try:
        xx = float(x)
        return xx
    except:
        print x
        return -9999


#  合并 var 数据中有异常数据的列
def merge_m():
    f = '../data/data00.csv'
    data = pd.read_csv(f)

    da = pd.read_csv('../data/data0_drop.csv')

    d = pd.merge(da,data,on='userid')
    d.fillna(-9999,inplace=True)
    print d.head()
    print d.shape
    d.to_csv('../data/da.csv',index=None)

def split_data(dir='data0'):
    data = pd.read_csv("../data/{}.csv".format(dir))
    cc = [c for c in data.columns if 'var' in c]

    cu = ['userid'] + cc

    d = data[cu]
    data.drop(cc, axis=1, inplace=True)

    print data.head()
    print data.shape

    data.to_csv('../data/data0_drop.csv', index=None)

    print d.head()
    print d.shape

    d.to_csv('../data/data1.csv',index=None)

# 拆分后的数据集
def filla_var_null(dir='data1'):

    data = pd.read_csv("../data/{}.csv".format(dir))
    cc = list(data.columns)
    cc.remove('userid')
    for c in cc:
        data[c] = data[c].apply(lambda x:KK(x))

    f = '../data/data00.csv'
    print data.head()
    print data.shape
    data.to_csv(f,index=None)




def getDatas(dir='train_data_'):
    loan_data = pd.read_csv("../data/{}.csv".format(dir))

    loan_data = loan_data.fillna(-9999)
    loan_data.index = loan_data['userid']
    loan_data.drop('userid',axis=1,inplace=True)

    # overdue_train，这是我们模型所要拟合的目标
    target = pd.read_csv('../../pcredit/train/overdue_train.txt',
                         header=None)
    target.columns = ['userid', 'label']
    target.index = target['userid']
    target.drop('userid',axis=1,inplace=True)
    # 构建模型
    # 分开训练集、测试集
    train = loan_data.iloc[0: 55596, :]
    test = loan_data.iloc[55596:, :]
    del loan_data
    return train,target,test

def getDatasR(dir='train_data_'):
    loan_data = pd.read_csv("../data/{}.csv".format(dir))

    loan_data.index = loan_data['userid']
    loan_data.drop('userid',axis=1,inplace=True)

    # overdue_train，这是我们模型所要拟合的目标
    target = pd.read_csv('../../pcredit/train/overdue_train.txt',
                         header=None)
    target.columns = ['userid', 'label']
    target.index = target['userid']
    target.drop('userid',axis=1,inplace=True)
    # 构建模型
    # 分开训练集、测试集
    train = loan_data.iloc[0: 55596, :]
    test = loan_data.iloc[55596:, :]

    from sklearn.cross_validation import train_test_split

    train_X, test_X, train_y, test_y = train_test_split(train,
                                                        target.label,
                                                        test_size=0.2,
                                                        random_state=0)

    return train_X, test_X, train_y, test_y,test


def getDatas2(dir='train_data_',k=0.2):
    train, target, test = getDatas(dir)
    from sklearn import metrics

    from sklearn.cross_validation import train_test_split

    ind_train = np.where(target > 0.5)[0]  # 获得训练数据为 1 的行
    label = target.iloc[ind_train]

    trainX = train.iloc[ind_train]
    ind_train0 = np.where(target < 0.5)[0]  # 获得训练数据为 1 的行
    label0 = target.iloc[ind_train0]
    trainX0 = train.iloc[ind_train0]

    del train

    train_X, test_X, train_y, test_y = train_test_split(trainX,
                                                        label,
                                                        test_size=k,
                                                        random_state=0)
    del trainX,label
    train_X0, test_X0, train_y0, test_y0 = train_test_split(trainX0,
                                                            label0,
                                                            test_size=k,
                                                            random_state=0)

    del trainX0,label0

    train_X = pd.concat([train_X, train_X0],axis=0)
    del train_X0
    train_X = train_X.sort_index()

    test_X = pd.concat([test_X, test_X0],axis=0)
    del test_X0
    test_X = test_X.sort_index()

    train_y = pd.concat([train_y, train_y0],axis=0)
    del train_y0
    train_y = train_y.sort_index()

    test_y = pd.concat([test_y, test_y0],axis=0)
    del test_y0
    test_y = test_y.sort_index()
    return train_X, test_X, train_y, test_y, test


def getDatas3(dir='train_data_'):
    train, target, test = getDatas(dir)
    from sklearn import metrics

    from sklearn.cross_validation import train_test_split

    ind_train = np.where(target > 0)[0]  # 获得训练数据为 1 的行

    label = target.iloc[ind_train]
    # print label

    trainX = train.iloc[ind_train]

    ind_train0 = np.where(target == 0)[0]  # 获得训练数据为 1 的行
    label0 = target.iloc[ind_train0]
    trainX0 = train.iloc[ind_train0]

    train_X, test_X, train_y, test_y = train_test_split(trainX,
                                                        label,
                                                        test_size=0.2,
                                                        random_state=0)

    train_X0, test_X0, train_y0, test_y0 = train_test_split(trainX0,
                                                            label0,
                                                            test_size=0.2,
                                                            random_state=0)
    train_X = pd.concat([train_X, train_X0])
    train_X = train_X.sort_index()

    test_X = pd.concat([test_X, test_X0])
    test_X = test_X.sort_index()

    train_y = pd.concat([train_y, train_y0])
    train_y = train_y.sort_index()

    test_y = pd.concat([test_y, test_y0])
    test_y = test_y.sort_index()
    return train_X, test_X, train_y, test_y, test


def getXGBoostDatas(dir='train_data'):
    loan_data = pd.read_csv("../data/{}.csv".format(dir))

    #loan_data.index = loan_data['userid']
    #loan_data.drop('userid',axis=1,inplace=True)

    # overdue_train，这是我们模型所要拟合的目标
    target = pd.read_csv('../../pcredit/train/overdue_train.txt',
                         header=None)
    target.columns = ['userid', 'label']
    #target.index = target['userid']
    #target.drop('userid',axis=1,inplace=True)
    # 构建模型
    # 分开训练集、测试集
    train = loan_data.iloc[0: 55596, :]
    test = loan_data.iloc[55596:, :]
    return train,target,test


if __name__=='__main__':
    merge_m()


"""
dir='/train/data_all'
loan_data = pd.read_csv("../data/{}.csv".format(dir))
loan_data = loan_data.fillna(0)


loan_data.rename(columns={'user_id': 'userid'}, inplace=True)
cols = loan_data.columns
loan_data.drop(cols[1],axis=1,inplace=True)
loan_data.index = loan_data['userid']
loan_data.drop('userid',axis=1,inplace=True)



def XX(x):
    try:
        f = float(x)
        return f
    except ValueError:
        print x
        return 0

def kk(col):
    loan_data[col] = loan_data[col].apply(lambda x:XX(x))
    print "col: ",col
    del loan_data[col]

#  woe 数据
def getDataSplit(dir='/train/data_all'):

    from multiprocessing import Pool
    #pool = Pool(3)

    #cols = loan_data.columns
    #c =cols[:3]
    c = ["var_bill_detail_var_time_bill",
         "max_bill_detail_var_time_bill",
         "mean_bill_detail_var_time_bill",
         "median_bill_detail_var_time_bill",
         "min_bill_detail_var_time_bill","sum_bill_detail_var_time_bill"]
    loan_data.drop(c,axis=1,inplace=True)
    #rs = [c for c in cols if 'var' in c ]
    #print rs
    #pool.map(kk,rs)
    #pool.close()
    #pool.join()

    loan_data.to_csv('../data/ddd.csv')
    print loan_data.head(1)
    print loan_data.shape


"""