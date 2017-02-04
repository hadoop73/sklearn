# coding:utf-8



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def merge(user='',bill=''):
    file_user = '../data/data_new1.csv'  # manman 数据
    #file_user = "../data/train/user_data_dummy.csv"
    file_bank = "../data/train/bank_all_data.csv"
    file_bill = "../data/train/bill_all_data.csv"
    file_browser = "../data/train/browser_all.csv"

    user_info = pd.read_csv(file_user)
    print 'user info:', user_info.shape

    bank_data = pd.read_csv(file_bank)
    print 'bank data:', bank_data.shape

    bill_data = pd.read_csv(file_bill)
    print 'bill data:', bill_data.shape

    browse_data = pd.read_csv(file_browser)
    print 'browse data:', browse_data.shape

    data = pd.merge(user_info,bank_data,on='userid',how='outer')
    data1 = pd.merge(bill_data,browse_data,on='userid',how='outer')
    data = pd.merge(data,data1,on='userid',how='outer')

    file_result = '../data/all_data1.csv'
    print "datas produced!"
    print "\t",file_result
    data.fillna(-9999,inplace=True)
    with open('../model/featurescore/amerge.txt', 'a') as f:
        s = """
../data/data_new1.csv
../data/train/bank_all_data.csv
../data/train/bill_all_data.csv
../data/train/browser_all.csv
最后合并的训练文件：../data/all_data1.csv
"""
        f.writelines(s)
    print data.head()
    print "datas size: ", data.shape

    data.to_csv(file_result,index=None)


# 0.42
def merge_b(path=""):
    # 0.42
    #file_result = '../data/data_new0.csv'  # manman 数据
    #data_file = "../data/data_brow.csv"
    data_file = "../data/train/user_data_dummy.csv"
    datas = pd.read_csv(data_file)
    #datas.rename(columns={'user_id':'userid'},inplace=True)

    datas.set_index('userid', inplace=True)

    file_bank = "../data/train/bank_detail_stage.csv"
    file_bill = "../data/train/bill_detail.csv"
    #file_browser = "../data/train/browse_history.csv"  # browser_history_new
    file_browser = "../data/train/browser_history_a.csv"
    bank_data = pd.read_csv(file_bank)
    bank_data.set_index('userid', inplace=True)

    datas = datas.join(bank_data)
    del bank_data

    file_browser5 = "../data/train/browser_history_5.csv" # 分段
    browser_history_5 = pd.read_csv(file_browser5)
    browser_history_5.set_index('userid', inplace=True)

    datas = datas.join(browser_history_5)
    del browser_history_5

    bill_data = pd.read_csv(file_bill)
    bill_data.set_index('userid', inplace=True)
    datas = datas.join(bill_data)
    del bill_data

    bill_data_stage = pd.read_csv('../data/train/bill_detail_stage.csv')
    bill_data_stage.set_index('userid', inplace=True)
    datas = datas.join(bill_data_stage)
    del bill_data_stage

    f_stage_2 = '../data/train/bill_detail_ad.csv'  # loan 时间 前后的数据
    bill_data_f_stage_2 = pd.read_csv(f_stage_2)
    bill_data_f_stage_2.set_index('userid', inplace=True)

    l = bill_data_f_stage_2.shape[1]
    bill_data_f_stage_2.columns = ["bill_detail_ad#" + str(i) for i in range(l)]

    datas = datas.join(bill_data_f_stage_2)
    del bill_data_f_stage_2

    #bill_data_stage5 = pd.read_csv('../data/train/bill_detail_time_stage5.csv')
    bill_data_stage5 = pd.read_csv('../data/train/bill_detail_time_stage52.csv') # 加了时间
    bill_data_stage5.set_index('userid', inplace=True)
    #  列名有冲突重新命名
    l = bill_data_stage5.shape[1]
    bill_data_stage5.columns = [ "bill_data_stage#"+str(i) for i in range(l)]
    datas = datas.join(bill_data_stage5)
    del bill_data_stage5

    browse_data = pd.read_csv(file_browser)
    browse_data.set_index('userid',inplace=True)
    l = browse_data.shape[1]
    browse_data.columns = ["browse_data_stage#" + str(i) for i in range(l)]
    datas = datas.join(browse_data)
    del browse_data

    #datas = datas.fillna(0)

    #datas = dummyTranform(datas,cols=['gender','career','education','marital'])

    print datas.head()
    # 0.42
    #file_result = '../data/train_data_new_dummy.csv'
    # 43
    #file_result = '../data/train_data_12.csv'
    file_result = '../data/{}.csv'.format(path)
    with open('../record/ab.txt', 'a') as f:
        f.writelines("\n\n总的合成数据：{}".format(path))
    print "datas produced!"
    print "\t",file_result
    print "datas size: ", datas.shape
    #datas.to_csv(file_result)
    #return path
    return datas



#  添加了 knn pca
def merg_c():
    file_result = '../data/train_data_new.csv'
    datas = pd.read_csv(file_result)
    datas.set_index('userid',inplace=True)
    pfile = '../data/train/knn_pca1.csv'
    p = pd.read_csv(pfile)
    p.set_index('userid', inplace=True)

    datas = datas.join(p)
    datas = dummyTranform(datas)
    print datas.head()

    f_result = '../data/train_data_new_kp_dummy.csv'
    datas.to_csv(f_result)

def merge_d():
    f = "../data/train_data_new.csv"
    datas = pd.read_csv(f)
    datas.set_index('userid', inplace=True)
    pfile = '../data/train/knn_pca1.csv'
    p = pd.read_csv(pfile)
    p.set_index('userid', inplace=True)

    datas = datas.join(p)
    datas = dummyTranform(datas,cols=['km2',
                   'km3','km4','km5','km6','km7','km8','km9','km10',])
    print datas.head()

    f_result = '../data/train_data_new2_knnpca_dummy.csv'
    datas.to_csv(f_result)

def merge_e(dir="",path=""):
    f = "../data/{}.csv".format(dir)
    datas = pd.read_csv(f)
    datas.set_index('userid', inplace=True)
    pfile = '../data/train/knn_pca2.csv'
    p = pd.read_csv(pfile)
    p.set_index('userid', inplace=True)

    datas = datas.join(p)
    datas = dummyTranform(datas,cols=['km4',
                   'km5','km6','km7','km8',
                    'km9','km10','km11','km12','km13','km14','km15'])

    print datas.head()

    f_result = '../data/{}.csv'.format(path)
    datas.to_csv(f_result)

def dummyTranform(datas,cols=['gender','career','education','marital','km2',
                   'km3','km4','km5','km6','km7','km8','km9','km10',]):
    for col in cols:
        datas[col].astype('category')
        d = pd.get_dummies(datas[col])
        d = d.add_prefix("{}#".format(col))
        datas = datas.join(d)
        datas.drop(col,axis = 1,inplace = True)
    return datas

def decode_hot(datas,path=""):
    #f = "../data/{}.csv".format(dir)
    #datas = pd.read_csv(f)
    #datas.set_index('userid', inplace=True)

    datas = dummyTranform(datas, cols=['gender','career','education','marital'])
    print datas.head()
    p = "../data/{}.csv".format(path)
    print datas.head()
    print datas.shape
    with open('../record/ab.txt', 'a') as f:
        f.writelines("\n编码数据：{}".format(path))
    datas.to_csv(p)

def merge_aa():
    col = ['gender','career','education','marital','residenceRegistType']
    file_result = '../data/data_new0.csv'
    d = pd.read_csv(file_result)
    d.drop(col,inplace=True,axis=1)

    file_user = "../data/train/user_data_dummy.csv"
    user_d = pd.read_csv(file_user)
    data = pd.merge(d,user_d,on='userid')
    print data.head()
    print data.shape
    data.to_csv('../data/data_new1.csv',index=None)

def merge_a(user='',bill=''):
    train_file = "../data/train.csv"
    test_file = "../data/test.csv"

    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    datas = pd.concat([train,test],axis=0)

    datas.rename(columns={'user_id':'userid'},inplace=True)

    datas.drop(['isOverdue'],axis=1,inplace=True)

    # 用于 merge_b 0.42
    file_result = '../data/data_new0.csv'
    datas.to_csv(file_result,index=None)
    print datas.head()
    print "\t", file_result
    print "datas size: ", datas.shape


def merge_br(d="",path=""):
    #file_result = '../data/{}.csv'.format(dir)
    #d = pd.read_csv(file_result)
    d.replace(np.NaN,-9999,inplace=True)
    #d.set_index('userid',inplace=True)

    #file_brow2 = '../data/train/browse_history_split2.csv'
    file_brow2 = '../data/train/bank_detail_split2.csv'
    browse_2 = pd.read_csv(file_brow2)
    browse_2.replace(np.NaN,-9999,inplace=True)
    browse_2.set_index('userid', inplace=True)

    d = d.join(browse_2)
    d.fillna(-9999,inplace=True)
    print d.head()
    print d.shape
    #fpath = '../data/train_12_bank2.csv'
    fpath = '../data/{}.csv'.format(path)  # bill stage  重新运行，修改了一些大的异常值
    with open('../record/ab.txt', 'a') as f:
        f.writelines("\n添加银行二分数据数据：{}".format(path))
    #d.to_csv(fpath)
    #return path
    return d

from predict_risk.model.GBDT import gbdt

# 添加其他模型的预测值
def merge_dg(data,path=""):
    # 其中数据 0，nan 被 -9999替换了
    fpath = '../data/{}.csv'.format(dir)
    file_dg = '../data/train/dg.csv'
    dg = pd.read_csv(file_dg)
    dg.rename(columns={'user_id':'userid'},inplace=True)
    dg.set_index('userid',inplace=True)

    #data = pd.read_csv(fpath)
    #tran_bank_2.set_index('userid',inplace=True)

    train_bank_dg = data.join(dg)
    print train_bank_dg.head()
    print train_bank_dg.shape

    #ffpath = '../data/train_dg_bank2.csv'
    ffpath = '../data/{}.csv'.format(path)  # bill stage  重新运行，修改了一些大的异常值
    with open('../record/ab.txt', 'a') as f:
        f.writelines("\n添加其他模型数据：{}".format(path))
    train_bank_dg.to_csv(ffpath)
    #return path
    #return train_bank_dg


def ch_data():
    fpath = '../data/data3965.csv'
    data45 = pd.read_csv(fpath)
    data45.drop('label',axis=1,inplace=True)

    data45.rename(columns={'user_id':'userid'},inplace=True)
    print data45.head()
    print data45.shape

    p = '../data/data45.csv'
    data45.to_csv(p,index=None)

def fill_0():
    p = '../data/data45.csv'
    data45 = pd.read_csv(p)
    data45.replace([0, np.NaN], -9999, inplace=True)
    data45.fillna(-9999,inplace=True)

    print data45.head()
    print data45.shape

    f = '../data/data0.csv'
    data45.to_csv(f)


if __name__=='__main__':

    #data = merge_b(path='data_train_new5')  #  合并原始数据
    #d = merge_br(data,path='data_browser_2_new5')  # 合并浏览记录的，放款时间二分数据
    #data = merge_dg(d,path='data_browser_dg_u2') # 合并其他模型的数据
    #path = decode_hot(data, path="data_browser_dg_decode2n5")
    merge()
    #merge_e(dir='data_browser_dg_decode2',path='data_kn_pc') # 合并分类信息
    print "merge finished!!!!"





