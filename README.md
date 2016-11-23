
## Scikit-learn 中文手册

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



## 构建虚拟环境

[Python 虚拟环境：Virtualenv](1)

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

## Python 学习

[Python 学习之](3)


 


  [1]: ./images/1479741888781.jpg "1479741888781.jpg"
  [2]: http://liuzhijun.iteye.com/blog/1872241
  [3]: ./python
  [4]: https://coderwall.com/p/xdox9a/running-ipython-cleanly-inside-a-virtualenv