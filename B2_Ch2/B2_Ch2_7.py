# B2_Ch2_7.py

###############
# Prepared by Ran An, Wei Lu, and Feng Zhang
# Editor-in-chief: Weisheng Jiang, and Sheng Tu
# Book 2  |  Financial Risk Management with Python
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2021
###############

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

dims = 3
step_num = 500
move_mode = [-1, 0, 1]
origin = np.zeros((1, dims))

# random walk
step_shape = (step_num, dims)
steps = np.random.choice(a=move_mode, size=step_shape)
path = np.concatenate([origin, steps]).cumsum(0)
start = path[:1]
stop = path[-1:]

# plot path
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(path[:,0], path[:,1], path[:,2],c='lightblue', marker='+');
ax.plot3D(start[:,0], start[:,1], start[:,2],c='green', marker='s')
ax.plot3D(stop[:,0], stop[:,1], stop[:,2],c='red', marker='o')
ax.set_title('Random Walk in '+str(dims)+'D')
