#-*- coding:utf-8 -*-


# 蒙特卡罗估算 pi

import numpy as np

def mc_pi(n = 100):
    m = 0
    i = 0;
    while i < n:
        x,y = np.random.rand(2)
        if x**2 + y**2 < 1:
            m += 1
        i += 1
    pi = 4. * m/n
    res = {'total_point':n,'point_in_circle':m,'estimated_pi':pi}

    return res

print mc_pi(n=10000)



