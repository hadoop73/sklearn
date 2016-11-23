#-*- coding:utf-8 -*-


## Imports
from pyspark import SparkContext

from operator import add
import sys

import re

if __name__ == "__main__":

    #conf = SparkConf().setMaster("local").setAppName("My App")
    sc = SparkContext(appName="PythonWordCount")
    lines = sc.textFile('data/shakespeare.txt')
    #re.split(r'(?:,|;|\s)\s*', line)  #x.split(' ')) \
    counts = lines.flatMap(lambda x: re.split(r'(?:\W)\W*',x)) \
                  .map(lambda x: (x,1)) \
                  . reduceByKey(add)
    output = counts.collect()
    for (word,count) in output:
        print "%s: %i"  %(word,count)

    sc.stop()




































