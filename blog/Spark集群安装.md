#  Spark 集群安装


[Spark 1.6.1分布式集群环境搭建](https://my.oschina.net/jackieyeah/blog/659741)

[Hadoop 2.6.4分布式集群环境搭建](https://my.oschina.net/jackieyeah/blog/657750)

-  [Hadoop 2.6.4分布式集群环境搭建](#id1)


-  [Spark 集群搭建](#id2)


<h2 id='id1'>Hadoop 2.6.4分布式集群环境搭建</h2>

###  修改 /etc/host 文件

```
127.0.0.1       localhost
192.168.109.137 master
192.168.109.139 slave01
192.168.109.138 slave02
```

使用 ping 测试连通性


###  配置 ssh 无密码访问集群机器

```
ssh-keygen -t dsa -P '' -f ~/.ssh/id_dsa
cat ~/.ssh/id_dsa.pub >> ~/.ssh/authorized_keys
```
将 slave01 和 slave02 的公钥 id_dsa.pub 传给 master

```
scp ~/.ssh/id_dsa.pub hadoop@master:/home/hadoop/.ssh/id_dsa.pub.slave01
scp ~/.ssh/id_dsa.pub hadoop@master:/home/hadoop/.ssh/id_dsa.pub.slave02
```
将 slave01 和 slave02 公钥信息追加到 authorized_keys 文件

```
cat id_dsa.pub.slave01 >> authorized_keys
cat id_dsa.pub.slave02 >> authorized_keys
```
将 master的公钥信息 authorized_keys 复制到 slave01 和 slave02 的 .ssh 目录下

```
scp authorized_keys hadoop@slave01:/home/hadoop/.ssh/authorized_keys
scp authorized_keys hadoop@slave02:/home/hadoop/.ssh/authorized_keys
```

使用 ssh slave01 测试连通性


###  集群配置

- hadoop-env.sh

```
export JAVA_HOME=/opt/java/jdk1.7.0_80
export HADOOP_PREFIX=/opt/hadoop-2.6.4
```
- core-site.xml

```
<configuration>
    <property>
        <name>fs.defaultFS</name>
        <value>hdfs://master:9000</value>
    </property>
    <property>
        <name>hadoop.tmp.dir</name>
        <value>/opt/hadoop-2.6.4/tmp</value>
    </property>
</configuration>
```

-  hdfs-site.xml

```
<configuration>
    <property>
        <name>dfs.replication</name>
        <value>3</value>
    </property>
</configuration>
```

-  mapred-site.xml

```
<configuration>
    <property>
        <name>mapreduce.framework.name</name>
        <value>yarn</value>
    </property>
</configuration>
```

- yarn-env.sh

```
export JAVA_HOME=/opt/java/jdk1.7.0_80
```

- slaves

```
master
slave01
slave02
```

slave01 和 slave02 同样配置

###  启动 Hadoop 集群

-  格式化 hdfs

在 master 上执行

```
hdfs namenode -format
```
-  启动 NameNode 和 DataNode

在 master 上执行 start-dfs.sh

jps 查看 Java 进程

![enter description here][1]

slave01 和 slave02 上同样可以看到 DataNode,奇怪的是在 http://master:50070 上看不到 slave 的 DataNode

![enter description here][2]

- 启动 ResourceManager 和 NodeManager

	在 master 上运行 start-yarn.sh

![enter description here][3]


Spark 集群搭建

**Spark 配置**

在 conf 目录,构建 spark-env.sh

```
cp spark-env.sh.template spark-env.sh
```

添加默认配置信息

```
export SCALA_HOME=/opt/scala-2.11.8
export JAVA_HOME=/opt/java/jdk1.7.0_80
export SPARK_MASTER_IP=192.168.109.137
export SPARK_WORKER_MEMORY=1g
export HADOOP_CONF_DIR=/opt/hadoop-2.6.4/etc/hadoop
```
在 slave 中添加

```
master
slave1
```

**启动 Spark 集群**

在 master 节点,运行 start-master.sh
在 worker 节点,运行 start-slaves.sh

运行 stop-master.sh 停止 master 节点,运行 stop-slaves.sh 启动所有 worker 节点



**DataNode 无法显示**

```
telnet master # 无法登录,需要关闭防火墙

sudo ufw disable  # 关闭,需要重启生效
sudo ufw status  # 显示状态 
``

  [1]: ./images/1481527861056.jpg "1481527861056.jpg"
  [2]: ./images/1481527939709.jpg "1481527939709.jpg"
  [3]: ./images/1481528011676.jpg "1481528011676.jpg"
