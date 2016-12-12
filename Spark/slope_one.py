# coding:utf-8


import numpy as np
import pandas as pd

'''
train_data = pd.read_csv("./data/train.csv")
test_data = pd.read_csv("./data/test.csv")



train_data.to_csv('../../Spark/data/train_slope.csv',index=None,header=None)

test_data.to_csv('../../Spark/data/test_slope.csv',index=None,header=None)

'''


from pyspark import SparkContext,SparkConf
import os
os.environ["SPARK_HOME"] = "/home/hadoop/spark-2.0.1-bin-hadoop2.7"   #KeyError: 'SPARK_HOME'

# master="spark://master:7077"

if __name__ == "__main__":
    sc = SparkContext(appName='SlopOne')


    def item_ij1(x):
        n = len(x)
        res = []
        for i in range(n):
            for j in range(i + 1, n):
                # res.append((x[i][0],x[j][0]))
                # res.append((x[j][0],x[i][0]))
                res.append(((x[i][0], x[j][0]), (1, int(x[i][1]) - int(x[j][1]))))
                res.append((str(x[j][0]).join(str(x[i][0])), (1, int(x[j][1]) - int(x[i][1]))))
        return res


    def xx(x):
        y = x.split(",")[0:3]
        return (int(y[0]), (int(y[1]), int(y[2])))


    train_rdd = sc.textFile('hdfs://master:9000/data/train_slope.csv')


    # .map(lambda x:x.split(',')[0:3]) \
    # .map(lambda x:(x[0],(x[1],x[2]))) \

    def kk(x):
        y = list(x)
        return (int(y[0]), (int(y[1]), int(y[2])))


    devij = train_rdd.map(lambda x: xx(x)) \
        .groupByKey().values() \
        .flatMap(lambda x: item_ij1(list(x))) \
        .reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))

    devij.saveAsTextFile("hdfs://master:9000/awc")










