# coding:utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
font = FontProperties(fname='/usr/share/fonts/truetype/arphic/ukai.ttc',size=14)


def getUserBankDatas():
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

    bank_data = pd.read_csv("../data/train/bank_data.csv")
    bank_data.index = bank_data['userid']
    bank_data.drop(['userid', 'bank_tag'],
                   axis=1,
                   inplace=True)

    # overdue_train，这是我们模型所要拟合的目标
    target = pd.read_csv('../../pcredit/train/overdue_train.txt', header=None)
    target.columns = ['userid', 'label']
    target.index = target['userid']
    target.drop('userid',
                axis=1,
                inplace=True)

    bank_data = user_info.join([bank_data])
    bank_data = bank_data.dropna()
    bank_data = bank_data.join([target])
    bank_data = bank_data.sort_index()

    # 构建模型
    # 分开训练集、测试集
    train = bank_data[bank_data.index < 55597]
    y = train.label
    train.drop(['label'], axis=1, inplace=True)
    test = bank_data[bank_data.index >= 55597]
    test.drop(['label'], axis=1, inplace=True)
    return train,y,test

















