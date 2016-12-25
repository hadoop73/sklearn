


[数据分析](#feature.ipynb)

数据中训练数据集: 1 - 55596

测试数据集: 55597 - 69495


##  哑变量处理

在线性回归模型中,有些类别用 0,1,2 来表示,它们之间的和是没有意义的;使用哑变量能够忽略这种影响

[实例](#feature.ipynb)

```
data[col].astype('category')  # 首先修改列的类型
dummy = pd.get_dummies(data[col])  #  获取哑变量数据
dummy = dummy.add_prefix('{}#'.format(col))  # 修改类名
data.drop(col,
           axis = 1,
           inplace = True)   # 删除原数据列
data = data.join(dummy)   # 添加到数据集合中
```






