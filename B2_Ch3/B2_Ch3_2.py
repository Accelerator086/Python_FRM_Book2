# B2_Ch3_2.py

###############
# Prepared by Ran An, Wei Lu, and Feng Zhang
# Editor-in-chief: Weisheng Jiang, and Sheng Tu
# Book 2  |  Financial Risk Management with Python
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2021
###############

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

## monte carlo
# number of rectangle
MC_num = 20

# integration domain
l = 0.
h = 5.

# function to be integrated
def func(x):
    return x**4 + x**3 + x**2


mpl.style.use('ggplot')
xx = np.linspace(l, h, 100)
plt.plot(xx, func(xx), color='r')
area_list = []
for _ in range(0, MC_num):
    
    # randomly generate mid point for rectangle
    x = l + (h-l)/MC_num *np.random.randint(1, MC_num-1)
    
    # height for each rectangle
    fvalue = func(x) 
    
    # area for each rectangle
    area = fvalue * (h - l)/MC_num 
    area_list.append(area)
    
    a = x-(h-l)/(2*MC_num)
    b = x+(h-l)/(2*MC_num)
    plt.plot([a,a], [0, func((a+b)/2)], color='#3C9DFF', alpha=0.5)
    plt.plot([b,b], [0, func((a+b)/2)], color='#3C9DFF', alpha=0.5)
    plt.plot([a,b], [func((a+b)/2), func((a+b)/2)], color='#3C9DFF', alpha=0.5)

intgr = sum(area_list)
print("Integration result: ", round(intgr))
 
plt.xlim([0,5])
plt.ylim([0,800])
plt.xlabel('x')
plt.ylabel('f(x)')
