# B2_Ch4_5.py

###############
# Prepared by Ran An, Wei Lu, and Feng Zhang
# Editor-in-chief: Weisheng Jiang, and Sheng Tu
# Book 2  |  Financial Risk Management with Python
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2021
###############

import matplotlib.pylab as plt
import numpy as np

x = np.arange(-8, 8, 0.1)
sig = 1 / (1 + np.exp(-x))
plt.plot(x, sig)
plt.title('Sigmoid Function')
plt.xlabel('x')
plt.ylabel('p')
plt.show()
