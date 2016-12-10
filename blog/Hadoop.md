
#  Hadoop

-    [HDFS 原理](#id1)

-    [MapReduce](#id2)


<h2 id='id1'>HDFS 原理</h2>

[Hadoop分布式文件系统HDFS的工作原理详述](http://www.36dsj.com/archives/40138)

[【Hadoop】HDFS的运行原理](http://www.cnblogs.com/laov/p/3434917.html)

HDFS 特点：

-  保存多个副本，提供容错机制，副本丢失或宕机自动恢复。默认3份

-  多台机器运行

-  适合大数据处理


![enter description here][1]

HDFS 是 Master 和 Slave 结构。分为 NameNode、SecondaryNameNode、DataNode三个角色

-  NameNode：管理数据块映射，处理客户端读写请求

-  SecondaryNameNode：保存 NameNode 的快照

-  DataNode：负责存储 Client 数据块，执行读写操作

**HDFS  写入**


![enter description here][2]


一个文件 100M 大小，一共三台机器 Rack1、Rack2、Rack3

-  Client 将 File 按 64M 分块为 block1 和 block2
-  Client  向 NameNode 发送写数据请求
-  NameNode 节点，记录 block 信息，并返回可用的 DataNode
-  Client 向 DataNode 发送 block1，block2
-  完成后，Client 向 NameNode 发送消息，通知写完了


**HDFS  读操作**

![enter description here][3]


-  Client 向 NameNode 发送读请求
-  NameNode 查看 Metadata 信息，返回 fileA 的 block 的位置
-  Client 顺序读取数据块


<h2 id='id2'>MapReduce</h2>

[MapReduce：详解 Shuffle 过程](http://langyu.iteye.com/blog/992916)




  [1]: ./images/1481378529262.jpg "1481378529262.jpg"
  [2]: ./images/1481379449644.jpg "1481379449644.jpg"
  [3]: ./images/1481379475064.jpg "1481379475064.jpg"