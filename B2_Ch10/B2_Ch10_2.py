# B2_Ch10_2.py

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

df = pd.read_csv(r'C:\Users\anran\Dropbox\FRM Book\CCR\EE.csv')

effective_ee = []
effective_ee.append(df['EE'].iloc[0]) 
for i in range(1, len(df.index)):
    effective_ee.append(max(effective_ee[i-1], df['EE'].iloc[i]))

# Effective EE    
df['Effective EE'] = effective_ee
# calculate EPE
epe = np.mean(df['EE'])
# calculate Effective EPE
effective_epe = np.mean(df['Effective EE'])

# plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df['Time'], df['EE'], '-o', label='EE')
ax.plot(df['Time'], df['Effective EE'], '-o', label='Effective EE')
ax.hlines(epe, 0, 1, linestyles ="dashed", colors ="#0070C0", label='EPE')
ax.hlines(effective_epe, 0, 1, linestyles ="dashed", colors ="#3C9DFF", label='Effective EPE')
ax.legend()

# add lables and title
ax.set_xlabel('Time')
ax.set_ylabel('Exposure')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.0])
ax.set_title('EE, Effective EE, EPE and Effective EPE')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
