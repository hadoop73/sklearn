
## 参考文章

[用Python开始机器学习（4：KNN分类算法）][1]


## matplotlib 中文乱码

**查找字体文件**

fc-match -v "AR PL UKai CN" | grep file

[Linux下python matplotlib.pyplot在图像上显示中文的问题][2]

[Python matplotlib画图的中文显示问题][3]

```python
from matplotlib.font_manager import FontProperties
font = FontProperties(fname='/usr/share/fonts/truetype/arphic/ukai.ttc',size=14)
plt.xlabel(u'身高',fontproperties=font)
plt.ylabel(u'体重',fontproperties=font)

```

## Error

```python
print(classification_report(y, answer, target_names=['thin', 'fat']))
```

计算正确率,召回率出错

ValueError: Mix type of y not allowed, got types set(['continuous', 'multiclass'])

因为两个参数有一个被视为连续性数据



 [1]: http://blog.csdn.net/lsldd/article/details/41357931
 [2]: http://blog.csdn.net/sinat_30071459/article/details/51694037
 [3]: http://blog.csdn.net/american199062/article/details/51690811
