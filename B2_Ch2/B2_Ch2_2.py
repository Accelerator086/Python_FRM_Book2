# B2_Ch2_2.py

###############
# Prepared by Ran An, Wei Lu, and Feng Zhang
# Editor-in-chief: Weisheng Jiang, and Sheng Tu
# Book 2  |  Financial Risk Management with Python
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2021
###############

import numpy as np

# today: stock price up
I = np.matrix([[1, 0]])

# transition matrix
T = np.matrix([[.75, 0.25],
               [.65, 0.35]])

n = 3
for i in range(0, n):
    T_tmp = I * T
    I = T_tmp
    # print ('The probability of stock price up/down after  day: ' , T)
    print ('The probability of stock price up/down after %d day: ' % (i+1), I)
