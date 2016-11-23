

# Spark 入门及应用

- [Spark shell 启动](#id1)
- [PySpark 环境配置](#id2)
- [WordsCount](#id3)

<h2 id="id1">Spark shell 启动</h2>
在 Spark 的安装目录下,bin/pyspark 运行 Python 版本的 Spark shell


**驱动程序**

每个 Spark 应用都由一个驱动器程序（ driver program）来发起集群上的各种
并行操作。驱动器程序包含应用的 main 函数，并且定义了集群上的分布式数据集，还对这
些分布式数据集应用了相关操作

驱动器程序通过一个 SparkContext 对象来访问 Spark。这个对象代表对计算集群的一个连
接。 shell 启动时已经自动创建了一个 SparkContext 对象，是一个叫作 sc 的变量


<h2 id="id2">PySpark 环境配置</h2>

- [python 下安装相关包](#h321)

- [Errors](#h322)

<h3 id="h321">python 下安装相关包</h3>

[PySpark 安装入门](1)

把 Spark 解压目录下的 pyspark 拷贝到 python 的包目录下

```
hadoop@master:~/.local/lib/python2.7/site-packages$ sudo cp -r ~/spark-2.0.1-bin-hadoop2.7/python/pyspark/ .
```

<h3 id="h322">Errors</h3>

py4j 错误
```
pip install py4j
```

<h2 id="id3">WordsCount</h2>

[Spark 下单词统计](pySpark.ipynb)

 [1]: http://blog.csdn.net/a819825294/article/details/51782773


































