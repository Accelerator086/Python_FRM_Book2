# B2_Ch3_3.py

###############
# Prepared by Ran An, Wei Lu, and Feng Zhang
# Editor-in-chief: Weisheng Jiang, and Sheng Tu
# Book 2  |  Financial Risk Management with Python
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2021
###############

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

MC_num = 2000

# integration domain
l = 0.
h = 5.

# function to be integrated
def func(x):
    return x**4 + x**3 + x**2

# plot function
X = np.linspace(l, h, 100)
plt.plot(X, func(X))

# rectangle region
y1 = 0
y2 = func(h)
area = (h-l)*(y2-y1)

underneath_list = []
x_list = []
y_list = []
for _ in range(MC_num):
    x = np.random.uniform(l,h,1)
    x_list.append(x)
    y = np.random.uniform(y1,y2,1)
    y_list.append(y)
    if abs(y)>abs(func(x)) or y<0:
        underneath_list.append(0)
    else:
        underneath_list.append(1)

# integration result
intgr = np.mean(underneath_list)*area
print("Integration result: ", round(intgr,2))

# visualize the process
df = pd.DataFrame()
df['x'] = x_list
df['y'] = y_list
df['underneath'] = underneath_list

plt.scatter(df[df['underneath']==0]['x'], df[df['underneath']==0]['y'], color='red')
plt.scatter(df[df['underneath']==1]['x'], df[df['underneath']==1]['y'], color='blue') 
plt.xlim([0,5])
plt.ylim([0,800])
plt.xlabel('x')
plt.ylabel('f(x)')
