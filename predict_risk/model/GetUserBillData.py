# coding:utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
font = FontProperties(fname='/usr/share/fonts/truetype/arphic/ukai.ttc',size=14)




def getUserBillData():
    # 用户id,性别,职业,教育程度,婚姻状态,户口类型
    names = ['userid', 'sex', 'job', 'edu', 'marriage', 'account']
    user_info_train = pd.read_csv("../../pcredit/train/user_info_train.txt", header=None)
    user_info_test = pd.read_csv("../../pcredit/test/user_info_test.txt", header=None)
    user_info = pd.concat([user_info_train, user_info_test])

    user_info.columns = names
    user_info.index = user_info['userid']
    user_info.drop('userid',
                   axis=1,
                   inplace=True)

    #  bill datas
    bill_datas = pd.read_csv('../data/train/bill_detail.csv')
    bill_datas.index = bill_datas['userid']
    bill_datas.drop('userid', axis=1, inplace=True)

    # overdue_train，这是我们模型所要拟合的目标
    target = pd.read_csv('../../pcredit/train/overdue_train.txt', header=None)
    target.columns = ['userid', 'label']
    target.index = target['userid']
    target.drop('userid',
                axis=1,
                inplace=True)

    user_bill = user_info.join([bill_datas])
    user_bill = user_bill.dropna()
    user_bill = user_bill.join([target])
    user_bill.drop(['bill_tag'], axis=1, inplace=True)
    user_bill = user_bill.sort_index()

    # 构建模型
    # 分开训练集、测试集
    train = user_bill[user_bill.index < 55597]
    y = train.label
    train.drop(['label'], axis=1, inplace=True)
    test = user_bill[user_bill.index >= 55597]
    test.drop(['label'], axis=1, inplace=True)

    return train,y,test
