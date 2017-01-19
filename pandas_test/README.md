

#  Pandas 数据处理工具学习


[Python数据处理：Pandas模块的 12 种实用技巧][1]

- [pandas 的数据结构](#h21)
- [操作](#h22)

<h2 id="h21">pandas 的数据结构</h2>

- [Series](#id1)

- [DataFrame](#id2)

- [索引对象](#id3)


<h3 id="id1">Series</h3>

类似一维数组的对象,由一组数据和一组相关的数据标签组成



```
obj = Series([4,7,-5,3])
obj1 = Series([4,7,-5,3],index=['d','b','a','c'])
```
接受字典创建,Series 算术运算会自动对其不同索引的数据


<h3 id="id2">DataFrame</h3>

表格型数据结构,含有一组有序的列,每列可以是不同的值类,DataFame 既有行索引也有列索引

```

```
- 选择列

    - df[['a','b']]

- 选择行

    - df.head(),默认5行;
    - df.head(3),选择3行
    - df.loc([0:6]),选择0-5行

- 删除列

    - df.drop(['a'])


- 为没有列名称的数据添加名字
    - pd.read_csv('../data',name=['a','b'])

- 重命名列名

    - df.rename(columns={'a':'ra','b':'rb'})


<h3 id="id3">索引对象</h3>
负责管理轴标签和其他元数据,Index对象不可以修改


<h2 id="h22">操作</h2>


<h3 id="h23">删除</h3>
删除某条轴上的项,默认删除行,用 axis=1 指定删除列

```
obj.drop(['d','c'],axis=1)
```

##  pandas

[读写操作以及数据处理](read_write.ipynb)


## 插入行，插入列

**插入列**

直接添加

```
d['c'] = d['a']-d['b'] # 根据 列a 和 列b 构建 列c
``

**插入行**

比较好的方法，所有的行都是放入数组的字典，直接用数组构建 DataFrame

```
r = []
r.append(d)  # r 中添加每一行，每行都是一个字典
df = pd.DataFrame(r) # 根据列表构建 DataFrame
```


方法二：在 DataFrame 中添加 DataFrame或列表 的形式，添加行
```
f = pd.DataFrame(columns=['a','b','c'])
f.append(df,ignore_index=True)  # 添加一个 DataFrame 的 df
```

添加列表

```
f = pd.DataFrame(columns=['a','b','c'])
f.loc[i] = [1,2,3]  # 以列表的形式添加一个行
```

 [1]: http://python.jobbole.com/85742/?utm_source=blog.jobbole.com&utm_medium=relatedPosts

