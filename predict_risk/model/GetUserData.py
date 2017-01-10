# coding:utf-8


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
font = FontProperties(fname='/usr/share/fonts/truetype/arphic/ukai.ttc',size=14)


def getUserData():
    # 用户id,性别,职业,教育程度,婚姻状态,户口类型
    names = ['userid', 'sex', 'job', 'edu', 'marriage', 'account']
    user_info_train = pd.read_csv("../../pcredit/train/user_info_train.txt", header=None)
    user_info_train.columns = names
    user_info_train.index = user_info_train['userid']
    user_info_train.drop('userid',
                         axis=1,
                         inplace=True)
    user_info_train = user_info_train.sort_index()

    # overdue_train，这是我们模型所要拟合的目标
    target = pd.read_csv('../../pcredit/train/overdue_train.txt', header=None)
    target.columns = ['userid', 'label']
    target.index = target['userid']
    target.drop('userid',
                axis=1,
                inplace=True)

    names = ['userid', 'sex', 'job', 'edu', 'marriage', 'account']
    user_info_test = pd.read_csv("../../pcredit/test/user_info_test.txt", header=None)
    user_info_test.columns = names

    user_info_test.index = user_info_test['userid']
    user_info_test = user_info_test.sort_index()
    user_info_test.drop('userid',
                        axis=1,
                        inplace=True)

    return user_info_train,target,user_info_test



















