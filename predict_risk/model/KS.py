# coding:utf-8




def KKS(pred=[0.8, 0.2, 0.7, 0.3, 0.5, 0.5, 0.5, 0.7, 0.7, 0.2],
        y=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]):
    import pandas as pd
    import warnings
    warnings.filterwarnings('ignore')
    d = {}
    d['pred'] = pred
    d['label'] = y
    df = pd.DataFrame(d)
    df.sort('pred',inplace=True)

    #print df

    s0 = df[df.label<0.5].shape[0]
    s1 = df[df.label>0.5].shape[0]

    k0,k1 = 0,0
    ans = 0
    k = 0
    for index,row in df.iterrows():
        i,j = row.label,row.pred
        if i==0:
            k0 += 1
        else:
            k1 += 1
        if abs(1.0*k0/s0-1.0*k1/s1) > ans:
            ans = abs(1.0*k0/s0-1.0*k1/s1)
            k = j
    return ans,k


def KS(pred=[0.8, 0.2, 0.7, 0.3, 0.5, 0.5, 0.5, 0.7, 0.7, 0.2],
        y=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]):
    import pandas as pd
    import warnings
    warnings.filterwarnings('ignore')
    d = {}
    d['pred'] = pred
    d['label'] = y
    df = pd.DataFrame(d)
    df.sort('pred',inplace=True)

    #print df
    df['label'] = df['label'].astype('int')
    s0 = df[df.label==0].shape[0]
    s1 = df[df.label==1].shape[0]
    if s0==0 or s1==0:
        return 0
    k0,k1 = 0,0
    ans = 0
    for index,row in df.iterrows():
            i,j = row.label,row.pred
            if i==0:
                k0 += 1
            else:
                k1 += 1
            if abs(1.0*k0/s0-1.0*k1/s1) > ans:
                ans = abs(1.0*k0/s0-1.0*k1/s1)
    return ans


import pandas as pd
import numpy as np

def y_ks(kk=696):
    y = pd.read_csv("../data/y/{}.csv".format(str(kk)))
    ks,k = KKS(y['prob'],y['label'])
    a =  y[(y['prob']>k)&(y['label']==0)]
    aa = y[y['label']==0]
    print a.head()
    print a.shape
    print aa.shape
    print "0 的错分率：　",1.0*a.shape[0] / aa.shape[0]

    b =  y[(y['prob'] <= k) & (y['label'] == 1)]
    bb = y[y['label'] == 1]
    print b.head()
    print bb.shape
    print "1 的错分率：　", 1.0 * b.shape[0] / bb.shape[0]

if __name__=="__main__":
    y = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    pred = [0.8, 0.2, 0.7, 0.3, 0.5, 0.5, 0.5, 0.7, 0.7, 0.2]
    #y_ks()
    print KS(pred=pred,y=y)
    #print KKS(pred,y)