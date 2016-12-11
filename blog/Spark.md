
#  Spark

-  [RDD 理解](#id1)

-  [Spark 优化](#id2)

-  [Spark 架构](#id3)


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



Spark  优化

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



Spark 架构


Spark 集群采用主从结构,一个节点负责中央协调,协调各个分布式工作节点.中央节点称为**驱动器**,工作节点称为**执行器**,驱动器和工作节点被称为**Spark 应用**


![enter description here][4]


**Spark 应用**通过**集群管理器**在集群中启动,Spark 自带的集群管理器被称为**独立集群管理器**


**驱动器**

-  把用户程序转为任务

	驱动器运行时,把逻辑图 DAG 转为物理执行计划;Spark 将多个操作合并到一个 Stage 中,每个 Stage 由多个任务组成,这些任务被打包发送到集群中

-  为执行器节点调度任务

	执行器启动后,会向驱动进程注册自己.每个执行器节点代表一个处理任务或存储 RDD 数据的进场
	Spark 驱动器根据当前的执行器节点集合,把任务基于数据位置分配给合适的执行器进场;执行完了会把数据缓存起来

Spark 应用的运行时信息通过网页界面呈现,默认端口 4040 (集群管理器网页界面:http://master:8080)


**执行器节点**

Spark 执行器是一种工作进程,负责在 Spark 作业中运行任务,任务相互独立

-  负责 Spark 应用的任务,并把结果返回给驱动器进场

-  自身的块管理,为用户程序的缓存提供内存式存储


**集群管理器**

Spark 依赖集群管理器来启动执行器节点


spark-submit 将应用提交到集群管理器上.并负责连接到相应的集群管理器上


**总结**

-  spark-submit 提交应用
-  spark-submit 脚本启动驱动器程序,调用用户定义的 main() 方法
-  驱动程序与集群管理器通信,申请资源以启动执行器节点
-  集群管理器为驱动器程序启动执行器节点
-  驱动器程序执行应用中的操作,并把工作以任务的形式发送到执行器进程
-  任务在执行器程序中进行计算并保存结果
-  驱动程序 main() 方法退出,或者调用 SparkContext.stop(),驱动器程序终止执行器进程,通过集群管理器释放资源

**Spark-submit**

如果在调用 Spark-submit 时,没有别的参数时,Spark 程序只会在本地执行.提交时可以将**集群地址**和**执行器进场大小**作为参数

```
bin/spark-submit --master spark://host:7077 --executor-memory 10g my_script.py
```
`--master` 标记集群 URL,spark:// 表示使用独立模式

选项参数可以分为两类:

-  调度信息,比如希望为作业申请的资源

-  运行时依赖,--py-Files,--jars


spark-submit 运行通过 --conf pro=value 设置 SparkConf 配置选项

**资源分配**

-  每个执行器进程内存
	
	`--executor-memory` 参数配置,默认 1GB,最大内存(默认 8GB)

-  占用核心总数的最大值

	`--total-executorcores` 参数设置,可以在配置文件中设置 spark.cores.max 的值
	`--num-executors` 设置固定数量的执行器节点,默认值为 2
	`--executor-cores` 设置每个执行器进场从 YARN 中占用的核心数目


**Hadoop YARN**

设置指向 Hadoop 配置目录的环境变量,spark-submit 指向一个特殊的主节点 URL 提交作业即可

```
export HADOOP_CONF_DIR="..."
spark-submit --master yarn yourapp
```


























  [1]: ./images/1481375279569.jpg "1481375279569.jpg"
  [2]: ./images/1481375319591.jpg "1481375319591.jpg"
  [3]: ./images/1481375728240.jpg "1481375728240.jpg"
  [4]: ./images/1481465621739.jpg "1481465621739.jpg"
