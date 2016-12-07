
#  机器学习之路


- [Scikit-learn 中文手册](#id1)

- [构建虚拟环境](#id2)

- [Python 学习](#id3)

- [牛人 blog](#id4)

- [XGBOOST 安装使用](#id5)

<h2 id="id1">Scikit-learn 中文手册</h2>

http://sklearn.lzjqsdd.com/

1.scikit-learn 安装

http://scikit-learn.org/dev/install.html

```
pip install -U scikit-learn
```

安装 scikit-learn 之前需要安装 numpy

[在Ubuntu 14.04 64bit上安装numpy和matplotlib库](1)

```
sudo apt-get install python-numpy
sudo apt-get install python-scipy
sudo apt-get install python-matplotlib
```


2.scikit-learn 学习笔记

http://www.cnblogs.com/maybe2030/p/4583087.html

3.在Python中实现你自己的推荐系统

http://python.jobbole.com/85516/

pydotplus.graphviz.InvocationException: GraphViz's executables not found

4.重新安装 graphviz

http://www.graphviz.org/Download_linux_ubuntu.php

5.决策树案例

http://scikit-learn.org/stable/modules/tree.html#classification


6.安装 python-tk,与 matplotlib.pyplot 相关联

sudo apt-get install python-tk


**交叉校验,逻辑回归,线性回归**



<h2 id="id2">构建虚拟环境</h2>

```
# 首先安装 virtualenv
sudo pip install virtualenv
# 再用 virtualenv 创建版本
virtualenv ~/env
```

[Python 虚拟环境：Virtualenv](http://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000/001432712108300322c61f256c74803b43bfd65c6f8d0d0000)

利用 Pycharm 可以继承本地的环境

![enter description here][1]

**启动虚拟环境**
```
cd ~/env2.7
source ./bin/activate
```

**退出虚拟环境**
```
deactivate
```


[使用 virtualenvwrapper](http://blog.csdn.net/luckytanggu/article/details/51592091)

创建不同的虚拟环境，并进行管理;在 pycharm 中通过路经指定具体的python版本解释器

```
pip install virtualenvwrapper

```


**虚拟环境中运行 ipython**

先在虚拟环境中安装 ipython
```
pip install ipython
```

再对 ipython 启动进行设置

[Running iPython cleanly inside a virtualenv](4)

```
alias ipy="python -c 'import IPython; IPython.terminal.ipapp.launch_new_instance()'"
ipy notebook # 启动 ipython notebook
```

<h2 id="id3">Python 学习</h2>

[Python 学习之](3)


<h2 id="id4">牛人 blog</h2>

[Bryan__的专栏](http://blog.csdn.net/bryan__)

[止战。机器学习](http://www.cnblogs.com/zhizhan/tag/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/)



<h2 id="id5">XGBOOST 安装使用</h2>

[官网](https://xgboost.readthedocs.io/en/latest/build.html)

[XGBOOST installation](https://github.com/dmlc/xgboost/blob/master/doc/python/python_intro.md)

```
git clone --recursive https://github.com/dmlc/xgboost
cd xgboost; make -j4

```

To install XGBoost, do the following:

- Run make in the root directory of the project

- In the python-package directory, run

```
python setup.py install
```

  [1]: ./images/1479741888781.jpg "1479741888781.jpg"
  [2]: http://liuzhijun.iteye.com/blog/1872241
  [3]: ./python
  [4]: https://coderwall.com/p/xdox9a/running-ipython-cleanly-inside-a-virtualenv