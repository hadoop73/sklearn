#-*- coding:utf-8 -*-


## Imports
from pyspark import SparkContext

from operator import add
import sys

if __name__ == "__main__":

    #conf = SparkConf().setMaster("local").setAppName("My App")
    sc = SparkContext(appName="PythonWordCount")
    lines = sc.textFile('data/words.txt')
    counts = lines.flatMap(lambda x: x.split(' ')) \
                  .map(lambda x: (x,1)) \
                  . reduceByKey(add)
    output = counts.collect()
    for (word,count) in output:
        print "%s: %i"  %(word,count)

    sc.stop()




































