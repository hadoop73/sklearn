


[���ݷ���](#feature.ipynb)

������ѵ�����ݼ�: 1 - 55596

�������ݼ�: 55597 - 69495


##  �Ʊ�������

�����Իع�ģ����,��Щ����� 0,1,2 ����ʾ,����֮��ĺ���û�������;ʹ���Ʊ����ܹ���������Ӱ��

[ʵ��](#feature.ipynb)

```
data[col].astype('category')  # �����޸��е�����
dummy = pd.get_dummies(data[col])  #  ��ȡ�Ʊ�������
dummy = dummy.add_prefix('{}#'.format(col))  # �޸�����
data.drop(col,
           axis = 1,
           inplace = True)   # ɾ��ԭ������
data = data.join(dummy)   # ��ӵ����ݼ�����
```






