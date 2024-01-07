# B2_Ch10_4.py

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
interest_rate_vol = 0.005
confidence_level = 0.99
T = 25.0

# calcualte pfe
t = np.arange(0.0, T, 0.05)
irs_pfe = interest_rate_vol*np.sqrt(t)*(T-t)*norm.ppf(confidence_level)

# plot
plt.style.use('fast')
plt.plot(t, irs_pfe)
plt.xlabel("Time in years")
plt.ylabel("PFE")
plt.title("PFE Evolution -- Interest Rate Swap")
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().yaxis.set_ticks_position('left')
plt.gca().xaxis.set_ticks_position('bottom')
