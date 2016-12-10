
#  Spark

-  [RDD 理解](#id1)

-  [Spark 优化](#id2)




<h2 id="id1">RDD 理解</h2>

[ 那些年我们对Spark RDD的理解](http://blog.csdn.net/stark_summer/article/details/50218641)

RDD: 弹性分布式数据集，一个提供了一些算子、只读的数据集合;可以基于物理存储，和其他 RDD  转换创建

```python
from pyspark import SparkContext
import os
os.environ["SPARK_HOME"] = "/home/hadoop/spark-2.0.1-bin-hadoop2.7"
sc=SparkContext(appName='Test')

#  通过文件创建 RDD
lines = sc.textFile('data/words.txt')
```


**转换和行动**

RDD 包括：transformation 和 actions 操作;

transformations 操作：是一种惰性操作，并不会立即计算;

actions 操作：立即计算，并返回结果给程序

Transformations 类型操作如下：

![enter description here][1]

Action类型的操作：

![enter description here][2]

**RDD 实现原理**

每个 RDD 数据以 Block 保存在多台机器上，Executor启动 BlockManagerslave，管理一部分 Block;元数据由 Driver 节点的 BlockManagerMaster 保存。BlockManagerslave 生成 Block 后向 BlockManagerMaster 注册该 Block。

![enter description here][3]


**RDD dependency 与 DAG**

新的 RDD 依赖于原来的 RDD，这种依赖关系最终形成 DAG

-  窄依赖，只依赖父 RDD 一个或部分 Partition，包括 map 和 union
-  宽依赖，依赖父 RDD 的所有 Partion，包括 groupBy 和 join



<h2 id="id2">Spark  优化</h2>

[Spark性能优化指南——基础篇](https://zhuanlan.zhihu.com/p/21922826)

[Spark性能优化指南——高级篇](https://zhuanlan.zhihu.com/p/22024169)

-  避免创建重复 RDD

-  尽量复用同一个 RDD

```
# 错误做法
rdd1 = sc.testFile("")
rdd2 = rdd1.map(...)
rdd1.reduceByKey(..)
rdd2.map(..)

#  实际上 rdd2 只是 rdd1 的又一次变换，不需要重建
rdd1.reduceByKey(..)
rdd1.map(..) # 把 rdd2 和 第一次 map 写到一起
```
-  对多次使用的 RDD 进行持久化





  [1]: ./images/1481375279569.jpg "1481375279569.jpg"
  [2]: ./images/1481375319591.jpg "1481375319591.jpg"
  [3]: ./images/1481375728240.jpg "1481375728240.jpg"