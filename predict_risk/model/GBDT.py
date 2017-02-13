# coding:utf-8


from GetData import getDatas
from WriteDatas import writeDatas

#  都是 pandas 的 DataFrame

names = ['train_data_1','train_data_2','train_data_3']


import logging,sys

logger = logging.getLogger('GBDT')

# 指定 logger 输出格式
#logging.basicConfig(format='%(asctime)s %(levelname)-8s:%(message)s', level=logging.DEBUG)
logging.basicConfig(format='%(message)s', level=logging.INFO)

#formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')

# 文件日志
file_handler = logging.FileHandler("test.log")
#file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# 控制台日志
console_handler = logging.StreamHandler(sys.stdout)
#console_handler.formatter = formatter
# 为 logger 添加日志处理器
logger.addHandler(console_handler)
# 指定日志最低输出级别,默认为 WARN 级别
#logger.setLevel(logging.INFO)




def gbdt_a(n_estimators=300,rate=0.1,max_depth=5,rand_state=0,name='train_data_5'):

    train,target,test = getDatas(name)
    print "data :",name
    print train.shape
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import GradientBoostingRegressor


    clf = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=rate,
                max_depth=max_depth, random_state=rand_state, loss='ls')  # .fit(train, target)

    from ROC import ROC,ROC2,ROC3
    logger.info("Datas name: %s",name)
    logger.info("n_estimators= %s rate= %s max_depth= %s rand_state= %s",
                n_estimators,rate,max_depth,rand_state)

    (model, ks) = ROC(clf, train, target)
    result = model.predict(test)
    writeDatas(result, test, "{}".format(ks))

"""
    clf = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=rate,
                                    max_depth=max_depth, random_state=rand_state, loss='ls')  # .fit(train, target)
    (model, ks) = ROC2(clf, train, target)
    logger.debug("K-S: %s",ks)

    result = model.predict(test)
    writeDatas(result, test, "{}".format(ks))

    clf = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=rate,
                                    max_depth=max_depth, random_state=rand_state, loss='ls')  # .fit(train, target)
    (model, ks) = ROC3(clf, train, target)
    logger.debug("K-S: %s",ks)

    result = model.predict(test)
    writeDatas(result, test, "{}".format(ks))
"""

def gbdt(train,target,test,n):

    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import GradientBoostingRegressor


    clf = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1,subsample=0.4,
                max_depth=5, random_state=0, loss='ls')  # .fit(train, target)

    from ROC import ROC,ROC2
    print "delete ",-1*n," feature"
    (model,ks) = ROC(clf,train,target)

    #(model, ks) = ROC2(clf, train, target)

    result = model.predict(test)
    writeDatas(result, test, "bn{}".format(ks))

#print result

#writeDatas(result,test,"4")

import random

if  __name__ == '__main__':
    gbdt_a(300)


def  find_X():
    random_seed = range(1000,2000,20)
    rate = [i/1000.0 for i in range(100,200,2)]
    max_depth = [6,4,5,8]
    n_estimators = range(90,500,2)

    random.shuffle(random_seed)
    random.shuffle(rate)
    random.shuffle(max_depth)
    random.shuffle(n_estimators)

    for i in range(25):
        gbdt_a(n_estimators[i],rate[i],max_depth[i%4],random_seed[i])






