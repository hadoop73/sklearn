

"""

import numpy as np
import pandas as pd


d = pd.DataFrame({'a':[2,'ad',6],'b':[3,'nan',2]})

b = pd.DataFrame({'a':[1,3,5],'b':[3,4,2]})




def kk(x):
    try:
        x = float(x)
        return x
    except:
        return -9999

for c in d.columns:
    d[c] = d[c].apply(lambda x:kk(x))

d.fillna(-9999,inplace=True)
#d.set_index('a',inplace=True)

#b.set_index('a',inplace=True)

c = pd.concat([d,b],axis=0)

c.drop(['a'],axis=1,inplace=True)

#c.set_index('a',inplace=True)
#c = c.sort_index()
print c

"""




a = [1,2,3]

a.remove(1)
print a














