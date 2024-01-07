# B2_Ch10_6.py

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

ir_vol = 0.005
fx_vol = 0.05
correlation = 0.20
confidence_level = 0.99
T = 5

t = np.arange(0.0, T, 0.01)
irs_pfe = ir_vol*np.sqrt(t)*(T-t)*norm.ppf(confidence_level)
forward_pfe = fx_vol*np.sqrt(t)*norm.ppf(confidence_level)
ccs_pfe = np.sqrt(irs_pfe*irs_pfe+forward_pfe*forward_pfe+2.0*correlation*irs_pfe*forward_pfe)

plt.style.use('fast')
plt.plot(t, irs_pfe, c='lightblue', label='IRS')
plt.plot(t, forward_pfe, c='dodgerblue', label='FX Forward')
plt.plot(t, ccs_pfe, c='red', label='CCS')

plt.xlabel("Time in years")
plt.ylabel("PFE")
plt.title("PFE Evolution")
plt.legend()
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().yaxis.set_ticks_position('left')
plt.gca().xaxis.set_ticks_position('bottom')
