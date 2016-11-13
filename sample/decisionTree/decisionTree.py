#-*- coding:utf-8 -*-


import numpy as np
import scipy as sp
from sklearn import tree
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split

''' 数据读取 '''
data = []
lables = []
with open("data/1.txt") as ifile:
    for line in ifile:
        # 分隔符
        tokens = line.strip().split(' ')
        # 添加数据
        data.append([float(tk) for tk in tokens[:-1]])
        # 分类标签
        lables.append(tokens[-1])

x = np.array(data)
lables = np.array(lables)
y = np.zeros(lables.shape)

''' 标签转换为 0/1 '''
y[lables=='fat'] = 1


''' 拆分数据用于训练和测试 '''
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

''' 使用信息熵作为划分标准,进行决策树训练 '''
clf = tree.DecisionTreeClassifier(criterion='entropy')
print clf

clf.fit(x_train,y_train)

''' 把决策树结果写入文件 '''
with open('data/tree.dot','w') as f:
    f = tree.export_graphviz(clf,out_file=f)

''' 绘图写入 pdf '''
import pydotplus

# dot_data = StringIO()
dot_data = tree.export_graphviz(clf, out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("data/data.pdf")


''' 系数反映每个特征影响力,越大特征在分类中作业越大 '''
print clf.feature_importances_

''' 测试结果的打印 '''
ans = clf.predict(x_train)
print x_train
print ans

print x_train
print np.mean(ans==y_train)


''' 准确率与召回率 '''
precision,recall,thresholds = precision_recall_curve(y_train,clf.predict(x_train))
ans = clf.predict_proba(x)[:,1]
print ans
print classification_report(y,ans,target_names=['thin','fat'])
