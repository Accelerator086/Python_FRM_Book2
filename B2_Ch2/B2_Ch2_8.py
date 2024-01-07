# B2_Ch2_8.py

###############
# Prepared by Ran An, Wei Lu, and Feng Zhang
# Editor-in-chief: Weisheng Jiang, and Sheng Tu
# Book 2  |  Financial Risk Management with Python
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2021
###############

import numpy as np
import matplotlib.pyplot as plt

N = 1000
T = 10
sigma = 0.5
mu = 0.8
dt = T / float(N)
t = np.linspace(0.0, N*dt, N+1)
W0 = [0]

# simulate the increments by normal random variable generator
np.random.seed(666)
increments = np.random.normal(0, 1*np.sqrt(dt), N)
Wt1 = W0 + list(np.cumsum(increments))
Wt2 = sigma*np.array(Wt1)
Wt3 = mu*t + sigma*np.array(Wt1)
# plt.figure(figsize=(15,10))
plt.plot(t, Wt1, label='W(t)')
plt.plot(t, Wt2, label='$\sigma$W(t)')
plt.plot(t, Wt3, label='$\mu$t+$\sigma$W(t)')
plt.plot(t, mu*t, '--', label='$\mu$t')
plt.legend()
plt.xlabel('Time')
plt.ylabel('X(t)')

plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().yaxis.set_ticks_position('left')
plt.gca().xaxis.set_ticks_position('bottom')
