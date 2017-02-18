# coding:utf-8

import pandas as pd
from GetData import getDatas2,getDatas3,getDatas,getData12,getDatas4
from WriteDatas import writeDatas

# import packages
import pandas as pd
import numpy as np
import scipy.stats.stats as stats

# import data
data = pd.read_csv("/home/liuwensui/Documents/data/accepts.csv", sep=",", header=0)
"""
https://statcompute.wordpress.com/2012/12/08/monotonic-binning-with-python/

============================================================
   min_ltv  max_ltv  bad  total  bad_rate
0        0       83   88    884  0.099548
1       84       92  137    905  0.151381
2       93       98  175    851  0.205640
3       99      102  173    814  0.212531
4      103      108  194    821  0.236297
5      109      116  194    769  0.252276
6      117      176  235    793  0.296343
============================================================
   min_bureau_score  max_bureau_score  bad  total  bad_rate
0               443               630  325    747  0.435074
1               631               655  242    721  0.335645
2               656               676  173    721  0.239945
3               677               698  245   1059  0.231350
4               699               709   64    427  0.149883
5               710               732   73    712  0.102528
6               733               763   53    731  0.072503
7               764               848   21    719  0.029207
============================================================
   min_age_oldest_tr  max_age_oldest_tr  bad  total  bad_rate
0                  1                 59  319    987  0.323202
1                 60                108  235    975  0.241026
2                109                142  282   1199  0.235196
3                143                171  142    730  0.194521
4                172                250  125    976  0.128074
5                251                588   93    970  0.095876
============================================================
   min_tot_tr  max_tot_tr  bad  total  bad_rate
0           0           8  378   1351  0.279793
1           9          13  247   1025  0.240976
2          14          18  240   1185  0.202532
3          19          25  165   1126  0.146536
4          26          77  166   1150  0.144348
============================================================
   min_tot_income  max_tot_income  bad  total  bad_rate
0            0.00         2000.00  323   1217  0.265407
1         2002.00         2916.67  259   1153  0.224631
2         2919.00         4000.00  226   1150  0.196522
3         4001.00         5833.33  231   1186  0.194772
4         5833.34      8147166.66  157   1131  0.138815
"""

# define a binning function
def mono_bin(Y, X, n=20):
    # fill missings with median
    X2 = X.fillna(np.median(X))
    r = 0
    while np.abs(r) < 1:
        d1 = pd.DataFrame({"X": X2, "Y": Y, "Bucket": pd.qcut(X2, n)})
        d2 = d1.groupby('Bucket', as_index=True)
        r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
        n = n - 1
    d3 = pd.DataFrame(d2.min().X, columns=['min_' + X.name])
    d3['max_' + X.name] = d2.max().X
    d3[Y.name] = d2.sum().Y
    d3['total'] = d2.count().Y
    d3[Y.name + '_rate'] = d2.mean().Y
    d4 = (d3.sort_index(by='min_' + X.name)).reset_index(drop=True)
    print "=" * 60
    print d4





