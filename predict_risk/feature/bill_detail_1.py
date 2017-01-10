# coding:utf-8



import numpy as np
import pandas as pd
from pandas import DataFrame,Series

import matplotlib.pyplot as plt


import logging
import sys
logger = logging.getLogger('bill_detail')

formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')

# 文件日志
file_handler = logging.FileHandler("test.log")
file_handler.setFormatter(formatter)  # 可以通过setFormatter指定输出格式


# 控制台日志
console_handler = logging.StreamHandler(sys.stdout)
console_handler.formatter = formatter

# 为 logger 添加日志处理器
logger.addHandler(console_handler)
logger.addHandler(file_handler)
# 指定日志最低输出级别,默认为 WARN 级别
logger.setLevel(logging.INFO)

names = ["userid", "time", "bank_id", "pre_amount_of_bill", "pre_repayment", "credit_amount", \
         "amount_of_bill_left", "least_repayment", "consume_amount", "amount_of_bill", "adjust_amount", \
         "circ_interest", "avail_amount", "prepare_amount", "repayment_state"]

bill_train = pd.read_csv("../../pcredit/train/bill_detail_train.txt", header=None)
bill_test = pd.read_csv("../../pcredit/test/bill_detail_test.txt", header=None)

bill_data = pd.concat([bill_train, bill_test])
bill_data.columns = names

import warnings
import statsmodels.api as sm
warnings.filterwarnings('ignore')


# 至少还款金额 least_repayment
def dealLeast_repayment():
    print "dealLeast_repayment() ..."
    least_repayment = bill_data[['userid', 'bank_id', 'least_repayment']]
    # 删除缺省数据
    least_repayment = least_repayment[least_repayment['least_repayment'] != 0]
    least_repayment_g = least_repayment.groupby(['userid','bank_id'])
    df = pd.DataFrame(columns=('userid','p'))
    i = 0
    for k,group in least_repayment_g:
        userid = k[0]
        # 信用卡的记录数多于 4 个才处理,获得每张信用卡在 aic 评价下的阶数参数
        logger.info('dealLeast_repayment userid:{}  bank id:{}'.format(k[0], k[1]))
        logger.info('dealLeast_repayment group len:{}'.format(len(group)))
        if len(group) >4:
            data2Diff = group['least_repayment'].diff()
            temp = np.array(data2Diff)[1:]
            p = sm.tsa.arma_order_select_ic(temp,max_ar=6,max_ma=6,ic='aic')['aic_min_order']
            pp = "{}{}".format(p[0],p[1])
            df.loc[i] = [userid,pp]
            i += 1
    df['least_repaymentcount'] = 1
    #  每个 userid 的参数可能有多个相同的阶数
    dd = pd.pivot_table(df, index=['userid', 'p'], values=['least_repaymentcount'], aggfunc=sum)
    dd = dd.unstack()
    n = dd.shape[1]
    dd.columns = ['least_repayment##{}'.format(i) for i in range(n)]
    dd = dd.fillna(0)
    dd.to_csv("../../pcredit/train/dealLeast_repayment.csv")

    #return dd


#  上期还款金额 pre_repayment
def dealPre_repayment():
    print "dealPre_repayment() ..."
    pre_repayment = bill_data[['userid', 'bank_id', 'pre_repayment']]
    # 删除缺省数据
    pre_repayment = pre_repayment[pre_repayment['pre_repayment'] != 0]
    pre_repayment_g = pre_repayment.groupby(['userid','bank_id'])
    df = pd.DataFrame(columns=('userid','p'))
    i = 0
    for k,group in pre_repayment_g:
        userid = k[0]
        # 信用卡的记录数多于 4 个才处理,获得每张信用卡在 aic 评价下的阶数参数
        logger.info('dealPre_repayment userid:{}  bank id:{}'.format(k[0], k[1]))
        logger.info('dealPre_repayment group len:{}'.format(len(group)))
        if len(group) >4:
            data2Diff = group['pre_repayment'].diff()
            temp = np.array(data2Diff)[1:]
            p = sm.tsa.arma_order_select_ic(temp,max_ar=6,max_ma=6,ic='aic')['aic_min_order']
            pp = "{}{}".format(p[0],p[1])
            df.loc[i] = [userid,pp]
            i += 1
    df['pre_repaymentcount'] = 1
    #  每个 userid 的参数可能有多个相同的阶数
    dd = pd.pivot_table(df, index=['userid', 'p'], values=['pre_repaymentcount'], aggfunc=sum)
    dd = dd.unstack()
    n = dd.shape[1]
    dd.columns = ['pre_repayment##{}'.format(i) for i in range(n)]
    dd = dd.fillna(0)
    dd.to_csv("../../pcredit/train/dealPre_repayment.csv")

    #return dd

#  本期账单余额 amount_of_bill_left
def dealAmount_of_bill_left():
    print "dealAmount_of_bill_left() ..."
    amount_of_bill_left = bill_data[['userid', 'bank_id', 'amount_of_bill_left']]
    # 删除缺省数据
    amount_of_bill_left = amount_of_bill_left[amount_of_bill_left['amount_of_bill_left'] != 0]
    amount_of_bill_left_g = amount_of_bill_left.groupby(['userid','bank_id'])
    df = pd.DataFrame(columns=('userid','p'))
    i = 0
    for k,group in amount_of_bill_left_g:
        userid = k[0]
        # 信用卡的记录数多于 4 个才处理,获得每张信用卡在 aic 评价下的阶数参数
        logger.info('dealAmount_of_bill_left userid:{}  bank id:{}'.format(k[0], k[1]))
        logger.info('dealAmount_of_bill_left group len:{}'.format(len(group)))
        if len(group) >4:
            data2Diff = group['amount_of_bill_left'].diff()
            temp = np.array(data2Diff)[1:]
            p = sm.tsa.arma_order_select_ic(temp,max_ar=6,max_ma=6,ic='aic')['aic_min_order']
            pp = "{}{}".format(p[0],p[1])
            df.loc[i] = [userid,pp]
            i += 1
    df['amount_of_bill_leftcount'] = 1
    #  每个 userid 的参数可能有多个相同的阶数
    dd = pd.pivot_table(df, index=['userid', 'p'], values=['amount_of_bill_leftcount'], aggfunc=sum)
    dd = dd.unstack()
    n = dd.shape[1]
    dd.columns = ['amount_of_bill_left##{}'.format(i) for i in range(n)]
    dd = dd.fillna(0)
    dd.to_csv("../../pcredit/train/dealAmount_of_bill_left.csv")

    #return dd

# 可用金额
def dealAvail_amount():
    print "dealAvail_amount() ..."
    avail_amount = bill_data[['userid', 'bank_id', 'avail_amount']]
    # 删除缺省数据
    avail_amount = avail_amount[avail_amount['avail_amount'] != 0]
    avail_amount_g = avail_amount.groupby(['userid','bank_id'])
    df = pd.DataFrame(columns=('userid','p'))
    i = 0
    for k,group in avail_amount_g:
        userid = k[0]
        # 信用卡的记录数多于 4 个才处理,获得每张信用卡在 aic 评价下的阶数参数
        logger.info('dealAvail_amount userid:{}  bank id:{}'.format(k[0], k[1]))
        logger.info('dealAvail_amount group len:{}'.format(len(group)))
        if len(group) >4:
            data2Diff = group['avail_amount'].diff()
            temp = np.array(data2Diff)[1:]
            if len(temp) < 1: continue
            p = sm.tsa.arma_order_select_ic(temp,max_ar=6,max_ma=6,ic='aic')['aic_min_order']
            pp = "{}{}".format(p[0],p[1])
            df.loc[i] = [userid,pp]
            i += 1
    df['avail_amountcount'] = 1
    #  每个 userid 的参数可能有多个相同的阶数
    dd = pd.pivot_table(df, index=['userid', 'p'], values=['avail_amountcount'], aggfunc=sum)
    dd = dd.unstack()
    n = dd.shape[1]
    dd.columns = ['avail_amount##{}'.format(i) for i in range(n)]
    dd = dd.fillna(0)
    dd.to_csv("../../pcredit/train/dealAvail_amount.csv")
    #return dd


# 处理 上期账单金额 pre_amount_of_bill
def dealPre_amount_of_bill():
    print "dealPre_amount_of_bill() ..."
    pre_amount_of_bill = bill_data[['userid', 'bank_id', 'pre_amount_of_bill']]
    # 删除缺省数据
    pre_amount_of_bill = pre_amount_of_bill[pre_amount_of_bill['pre_amount_of_bill'] != 0]
    pre_amount_of_bill_g = pre_amount_of_bill.groupby(['userid','bank_id'])
    df = pd.DataFrame(columns=('userid','p'))
    i = 0
    for k,group in pre_amount_of_bill_g:
        userid = k[0]
        # 信用卡的记录数多于 4 个才处理,获得每张信用卡在 aic 评价下的阶数参数
        logger.info('dealPre_amount_of_bill userid:{}  bank id:{}'.format(k[0],k[1]))
        logger.info('dealPre_amount_of_bill group len:{}'.format(len(group)))
        if len(group) >4:
            data2Diff = group['pre_amount_of_bill'].diff()
            temp = np.array(data2Diff)[1:]
            p = sm.tsa.arma_order_select_ic(temp,max_ar=6,max_ma=6,ic='aic')['aic_min_order']
            pp = "{}{}".format(p[0],p[1])
            df.loc[i] = [userid,pp]
            i += 1
    df['pre_amount_of_billcount'] = 1
    #  每个 userid 的参数可能有多个相同的阶数
    dd = pd.pivot_table(df, index=['userid', 'p'], values=['pre_amount_of_billcount'], aggfunc=sum)
    dd = dd.unstack()
    n = dd.shape[1]
    dd.columns = ['pre_amount_of_bill##{}'.format(i) for i in range(n)]
    dd = dd.fillna(0)
    dd.to_csv("../../pcredit/train/dealPre_amount_of_bill.csv")
    #return dd


"""
方法1
import datetime
starttime = datetime.datetime.now()
#long running
endtime = datetime.datetime.now()
print (endtime - starttime).seconds
方法 2
start = time.time()
run_fun()
end = time.time()
print end-start
方法3
start = time.clock()
run_fun()
end = time.clock()
print end-start
方法1和方法2都包含了其他程序使用CPU的时间，是程序开始到程序结束的运行时间。
方法3算只计算了程序运行的CPU时间                                                                                                                                                                                                                                                                                                                                                                     
"""

#  多线程

import threading
threads = []

threads.extend([threading.Thread(target=dealPre_amount_of_bill),
               threading.Thread(target=dealAvail_amount),
               threading.Thread(target=dealAmount_of_bill_left),
               threading.Thread(target=dealPre_repayment),
               threading.Thread(target=dealLeast_repayment)])
if __name__=='__main__':
    import datetime
    starttime = datetime.datetime.now()

    '''
    pre_amount_of_bill = dealPre_amount_of_bill()
    avail_amount = dealAvail_amount()
    amount_of_bill_left = dealAmount_of_bill_left()
    pre_repayment = dealPre_repayment()
    least_repayment = dealLeast_repayment()
    '''
    for t in threads:
        t.start()
    #data = pre_amount_of_bill.join([avail_amount,amount_of_bill_left,pre_repayment,least_repayment])
   # data.to_csv("../../pcredit/train/bill_detail_data.csv")
    #print data.head()

    endtime = datetime.datetime.now()
    print (endtime - starttime).seconds