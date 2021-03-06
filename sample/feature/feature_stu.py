#-*- coding:utf-8 -*-


from sklearn.datasets import load_iris

'''
鸢尾花,包含四个特征: 花萼长度,花萼宽度,花瓣长度,花瓣宽度;特征都为正浮点型数据,单位厘米
目标分类由: 山鸢尾,杂色鸢尾,维吉尼亚鸢尾
'''


# 导入 iris 数据集
iris = load_iris()

# 特征矩阵
print iris.data

# 目标向量
print iris.target

print iris

from sklearn.preprocessing import StandardScaler

# 标准化数据,前提特征服从正太分布
StandardScaler().fit_transform(iris.data)


# 区间缩放法,返回值缩放到 [0,1] 之间
from sklearn.preprocessing import MinMaxScaler

MinMaxScaler().fit_transform(iris.data)



# 归一化,对样本数据行进行归一化,转化为 "单位向量" ,在点乘运算或其他核函数计算中,由统一标准

from sklearn.preprocessing import Normalizer

Normalizer().fit_transform(iris.data)


# 特征二值化,大于阀值赋值 1,小于阀值赋值 0

from sklearn.preprocessing import Binarizer

Binarizer(threshold=3).fit_transform(iris.data)


# 哑编码
from sklearn.preprocessing import OneHotEncoder

OneHotEncoder().fit_transform(iris.target.reshape((-1,1)))


from numpy import vstack, array, nan
from sklearn.preprocessing import Imputer

#缺失值计算，返回值为计算缺失值后的数据
#参数missing_value为缺失值的表示形式，默认为NaN
#参数strategy为缺失值填充方式，默认为mean（均值）
Imputer().fit_transform(vstack((array([nan, nan, nan, nan]), iris.data)))





from sklearn.preprocessing import PolynomialFeatures

#多项式转换
#参数degree为度，默认值为2
PolynomialFeatures().fit_transform(iris.data)



'''
  特征选择
'''

from sklearn.feature_selection import VarianceThreshold

#方差选择法，返回值为特征选择后的数据
#参数threshold为方差的阈值
VarianceThreshold(threshold=3).fit_transform(iris.data)



# 基于树模型的特征选择法
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier

#GBDT作为基模型的特征选择
SelectFromModel(GradientBoostingClassifier()).fit_transform(iris.data, iris.target)

