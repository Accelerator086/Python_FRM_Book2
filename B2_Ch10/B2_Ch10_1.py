# B2_Ch10_1.py

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
import scipy.stats as stats

# parameters
mu = 3.0 
sigma = 5.0
alpha = 0.97

# generate normal distribution
x1 = mu-20
x2 = mu+20
x = np.arange(x1, x2, 0.001)
y = norm.pdf(x, mu, sigma)

# calculate EE and PFE
EE = mu*stats.norm.cdf(mu/sigma) + sigma*stats.norm.pdf(mu/sigma)
PFE = mu + sigma*stats.norm.ppf(alpha)
print(' EE: ', round(EE,2)) 
print(' PFE: ', round(PFE,2))

# plot and identify EE PFE
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(x, y)

# fill exposure area
x0 = np.arange(0, x2, 0.001)
y0 = norm.pdf(x0, mu, sigma)
ax.fill_between(x0, y0, 0, color='moccasin')
ax.fill_between(x, y, 0, alpha=0.5, color='palegreen')

# draw vertical line to identify mu, EE and PFE
ax.vlines(mu, 0, norm.pdf(mu, mu, sigma), linestyles ="dashed", colors ="#B7DEE8", label='$\\mu$')
ax.vlines(EE, 0, norm.pdf(EE, mu, sigma), linestyles ="dashed", colors ="#0070C0", label='EE')
ax.vlines(PFE, 0, norm.pdf(PFE, mu, sigma), linestyles ="dashed", colors ="#3C9DFF", label='PFE')

# add x ticks to mu, EE and PFE
fig.canvas.draw()
labels = [w.get_text() for w in ax.get_xticklabels()]
locs = list(ax.get_xticks())
labels += ['$\\mu$', '$EE$', '$PFE$']
locs += [mu, EE, PFE]
ax.set_xticks(locs) 
ax.set_xticklabels(labels)

# add lables and title
ax.set_xlabel('MtM')
ax.set_ylabel('Probability')
ax.set_title('EE and PFE for a Normal Distribution')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
