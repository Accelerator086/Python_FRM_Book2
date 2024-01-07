# B2_Ch3_1.py

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

# number of rectangle
N = 20

# integration domain
l = 0.
h = 5.

# function to be integrated
def func(x):
    return x**4 + x**3 + x**2

x = np.linspace(l, h, N)

# height for each rectangle
fvalue = func(x) 

# area for each rectangle
area = fvalue * (h - l)/N 
intgr = sum(area)

print("Integration result: ", round(intgr))

mpl.style.use('ggplot')
plt.plot(x, func(x), color='r')
for i in range(1,len(x)):
    a = x[i-1]
    b = x[i]
    plt.plot([a,a], [0, func((a+b)/2)], color='#3C9DFF', alpha=0.5)
    plt.plot([b,b], [0, func(b)], color='#3C9DFF', alpha=0.5)
    plt.plot([a,b], [func((a+b)/2), func((a+b)/2)], color='#3C9DFF', alpha=0.5)

width = x[N-4]-x[N-5]
height =  0.5*(func(x[N-4])+func(x[N-5]))
Xi = 0.5*(x[N-5]+x[N-4])

rect = mpl.patches.Rectangle((x[N-5],0), width, height, color='#DBEEF4')
plt.gca().add_patch(rect)

# mark Xi
plt.annotate('Xi', xy=(Xi, 0), xytext=(Xi, 40), horizontalalignment='center', verticalalignment='center',
              arrowprops=dict(arrowstyle="-|>",
                              connectionstyle="arc3",
                              mutation_scale=10,
                              color='r',
                              fc="w"))
plt.text(Xi, 0.5*height, 'Ai', horizontalalignment='center', verticalalignment='center')

# mark width
plt.annotate(r'',
            xy=(x[N-5], height+10),
            xytext=(x[N-4], height+10),
            arrowprops=dict(arrowstyle="<|-|>",
                            connectionstyle="arc3",
                            mutation_scale=20,
                            color='coral',
                            fc="w")
            )
plt.text(Xi, height+20, '(h-l)/N', horizontalalignment='center', verticalalignment='center')

# mark height
plt.annotate(r'',
            xy=(x[N-4]+0.05, 0), #xycoords='data',
            xytext=(x[N-4]+0.05, height), #textcoords='data',
            arrowprops=dict(arrowstyle="<|-|>",
                            connectionstyle="arc3",
                            mutation_scale=20,
                            color='coral',
                            fc="w")
            )
plt.text(x[N-4]+0.1, height/2, 'f(Xi)', horizontalalignment='center', verticalalignment='center')

plt.xlim([0,5])
plt.ylim([0,800])
plt.xlabel('x')
plt.ylabel('f(x)')
