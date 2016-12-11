#  Spark 集群安装

- [Hadoop 2.6.4分布式集群环境搭建](#id1)




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



