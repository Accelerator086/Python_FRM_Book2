# B2_Ch10_3.py

###############
# Prepared by Ran An, Wei Lu, and Feng Zhang
# Editor-in-chief: Weisheng Jiang, and Sheng Tu
# Book 2  |  Financial Risk Management with Python
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2021
###############

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# parameters
drift = 0.05
vol = 0.10
confidence_level = 0.99
T = 5

# calculate pfe
t = np.arange(0.0, T, 0.01)
forward_pfe = drift*t+vol*np.sqrt(t)*norm.ppf(confidence_level)

# plot
plt.style.use('fast')
plt.plot(t, forward_pfe)
plt.grid(True)
plt.xlabel("Time in years")
plt.ylabel("PFE")
plt.title("PFE Evolution -- Forward")
plt.grid(None)  
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().yaxis.set_ticks_position('left')
plt.gca().xaxis.set_ticks_position('bottom')
