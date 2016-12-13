
#  Spark 算子

- groupByKey

    将 RDD[K,V] 中每个 K 对应的的 V 合并到一个集合
    
    [Spark算子：RDD键值转换操作(3)–groupByKey、reduceByKey](http://lxw1234.com/archives/2015/07/360.htm)


-  reduceByKey,flatMapValues(func)

    合并 RDD[K,V] 中每个 K 的 V
	
	对于 RDD {(1,2),(3,4),(3,6)}
	
	rdd.flatMapValues(x => (x to 5))
	
![enter description here][1]

-  combineByKey,foldByKey

      该函数用于将RDD[K,V]转换成RDD[K,C],这里的V类型和C类型可以相同也可以不同。
      由 3 个函数组成，第一个函数用来创建容器 C，第二个函数用于在每个partition进行迭代，在容器中添加元素
      第三个函数用于合并每个 partition
      
      [PairRDD中算子combineByKey图解](http://www.cnblogs.com/seaspring/p/5721853.html)


-  coalesce,repartition

    重新对 RDD 进行分区，coalesce 由两个参数，第一个参数表示分区的数目，第二个参数表示是否进行 shuffle，默认为 false
    repartition  是 coalesce 第二个参数为 true 的实现
    
    [Spark算子：RDD基本转换操作(2)–coalesce、repartition](http://lxw1234.com/archives/2015/07/341.htm)


-   reduce,fold,aggregate

	 reduce 接受一个函数为参数，这个函数操作两个 RDD 的元素类型数据并返回同样类型的新数据
	
```
# 总和
sum = rdd.reduce(lambda x,y : x+y)
```

	fold 和 reduce 返回值相同，多了一个初始值
	
	aggregate() 返回类型可以和 RDD 的类型不同

```
# 设置初始值给 acc，使用第一个函数在每个分区统计；再用第二个函数统计所有分区
sumCount = num.aggregate((0,0),
							(lambda acc,value: (acc[0]+value,acc[1]+1),
							(lambda acc1,acc2:(acc1[0]+acc2[0],acc1[1]+acc2[1]))))
							
return sumCount[0] / float(sumCount[1])
```

![enter description here][2]


  [1]: ./images/1481631103862.jpg "1481631103862.jpg"
  [2]: ./images/1481634525913.jpg "1481634525913.jpg"