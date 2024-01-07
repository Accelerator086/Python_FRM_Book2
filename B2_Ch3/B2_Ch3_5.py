# B2_Ch3_5.py

###############
# Prepared by Ran An, Wei Lu, and Feng Zhang
# Editor-in-chief: Weisheng Jiang, and Sheng Tu
# Book 2  |  Financial Risk Management with Python
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2021
###############

import matplotlib.pyplot as plt
import seaborn as sns
import random

l = 2
ax, ay = -l/2, l/2
bx, by = -l/2, -l/2
cx, cy = l/2, l/2
dx, dy = l/2, -l/2

ox, oy = int((ax+cx)/2), int((ay+by)/2)

point_num_list = [10, 50, 200, 500, 1000, 10000]

rows = 3
cols = 2
fig, ax = plt.subplots(rows, cols, figsize=(14,8))
fign = 0
fig_label = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

for i in range(rows):
    for j in range(cols):
        print('Figure #: ', [i, j])
        inside = 0
        for _ in range(point_num_list[fign]):
            x_inside = []
            y_inside = []
            x_outside = []
            y_outside = []
            
            x = random.uniform(-l/2, l/2)
            y = random.uniform(-l/2, l/2)
            if (x-ox)**2+(y-oy)**2 <= (l/2)**2:
                inside += 1
                x_inside.append(x)
                y_inside.append(y)
            else:
                x_outside.append(x)
                y_outside.append(y)
            
            sns.scatterplot(x=x_inside, y=y_inside, color='g', ax=ax[i, j])
            sns.scatterplot(x=x_outside, y=y_outside, color='r', ax=ax[i, j])
            ax[i, j].set_title(fig_label[fign], loc='left')
            ax[i, j].set_aspect('equal')
            ax[i, j].set_xticks([-1, 0, 1])
            ax[i, j].set_yticks([-1, 0, 1])
        pi = 4*inside/point_num_list[fign]
        print('Estimated /pi is %.4f based on %s points simulation.' %(pi,  point_num_list[fign]))            
        fign+=1
