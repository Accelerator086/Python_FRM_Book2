# B2_Ch2_5.py

###############
# Prepared by Ran An, Wei Lu, and Feng Zhang
# Editor-in-chief: Weisheng Jiang, and Sheng Tu
# Book 2  |  Financial Risk Management with Python
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2021
###############

import numpy as np
import matplotlib.pyplot as plt

# define parameters
dims = 1
step_num = 500
path_num = 10
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
    plt.plot(np.arange(step_num+1), path, marker='+', markersize=0.02);
    plt.plot(0, start, c='green', marker='s')
    plt.plot(step_num, stop, c='red', marker='o')
   
plt.title('Random Walk in 1D')
plt.xlabel('Step')
plt.ylabel('Position')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().yaxis.set_ticks_position('left')
plt.gca().xaxis.set_ticks_position('bottom')