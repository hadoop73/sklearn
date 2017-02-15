# coding:utf-8



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import metrics

import random

from predict_risk.model.KS import KS


target = pd.read_csv('../../pcredit/train/overdue_train.txt',
                         header=None)
target.columns = ['userid', 'label']
#train = loan_data.iloc[0: 55596, :]
#test = loan_data.iloc[55596:, :]
def bagging3():
    preds = pd.read_csv('data/train_test_pred.csv')
    cols = list(preds.columns)
    kks = 0
    auc = 0
    nn = np.arange(0,105,5)
    for i in range(1,len(cols)-2):
        for j in range(i+1,len(cols)-1):
            for k in range(j+1, len(cols)):
                for m in range(20):
                    wa = random.choice(nn)
                    wb = random.choice(nn)
                    if wa + wb > 100:
                        if wa > wb:
                            wa = wa - 50
                        else:
                            wb = wb - 50

                    wc = 100 - wa - wb
                    wa = 0.01 * wa
                    wb = 0.01 * wb
                    wc = 0.01 * wc
                    print wa,wb,wc
                    print preds[cols[i]].head()
                    print preds[cols[j]].head()
                    print preds[cols[k]].head()
                    preds['probability'] = wa*preds[cols[i]] + wb*preds[cols[j]] + wc*preds[cols[k]]

                    t = preds[['userid','probability']]
                    train = t.iloc[0: 55596]
                    print train.head()
                    test = t.iloc[55596:]
                    print test.head()

                    ks = KS(train['probability'],target.label)
                    fp, tp, thresholds = metrics.roc_curve(target.label, train['probability'], pos_label=1)
                    au = metrics.auc(fp, tp)
                    if ks >= kks or au >= auc:
                        kks = max(kks,ks)
                        auc = max(au,auc)
                        with open('data/bag.txt','a') as f:
                            f.writelines("\nKS: "+str(ks))
                            f.writelines("AUC: " + str(au))
                            f.writelines("组合列: " + cols[i] + " " + cols[j] + " " + cols[k])
                            f.writelines("权重: " + str(wa) + " " + str(wb) + " " + str(wc))
                            f.close()
                        test.to_csv('data/bag/{}.csv'.format(ks),index=None)




if __name__=='__main__':
    bagging3()











