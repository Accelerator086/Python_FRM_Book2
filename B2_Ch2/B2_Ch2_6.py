# B2_Ch2_6.py

###############
# Prepared by Ran An, Wei Lu, and Feng Zhang
# Editor-in-chief: Weisheng Jiang, and Sheng Tu
# Book 2  |  Financial Risk Management with Python
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2021
###############

import numpy as np
import matplotlib.pyplot as plt

dims = 2
step_num = 500
path_num = 1
move_mode = [-1, 1]
origin = np.zeros((1, dims))

for path in range(path_num):
    # random walk
    step_shape = (step_num, dims)
    steps = np.random.choice(a=move_mode, size=step_shape)
    path = np.concatenate([origin, steps]).cumsum(0)
    start = path[:1]
    stop = path[-1:]
    # plot path
    plt.plot(path[:,0], path[:,1], marker='+', markersize=0.02, c='lightblue');
    plt.plot(start[:,0], start[:,1], marker='s', c='green') 
    plt.plot(stop[:,0], stop[:,1], marker='o', c='red')

plt.title('Random Walk in '+str(dims)+'D')
plt.xlabel('x')
plt.ylabel('y')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().yaxis.set_ticks_position('left')
plt.gca().xaxis.set_ticks_position('bottom')
