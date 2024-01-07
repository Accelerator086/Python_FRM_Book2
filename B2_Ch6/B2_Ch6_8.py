# B2_Ch6_8.py

###############
# Prepared by Ran An, Wei Lu, and Feng Zhang
# Editor-in-chief: Weisheng Jiang, and Sheng Tu
# Book 2  |  Financial Risk Management with Python
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2021
###############

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

def cash_or_nothing_delta(S0, vol, r, q, t, K, Q, PutCall):
    d2 = (np.log(S0 / K) + (r - q - 0.5 * vol ** 2) * t) / (vol * np.sqrt(t))
    delta =  PutCall*Q*np.exp(-r * t) * norm.pdf(PutCall*d2, 0.0, 1.0) / (vol*S0*np.sqrt(t))    
    return delta

    # Inputs
r = 0.01  #risk-free interest rate
q = 0.0 # dividend yield
K = 55   #strike price
vol = 0.45 #volatility
t_base = 2.0
PutCall = 1 # 1 for call;-1 for put
spot = np.arange(20,95,0.5)  #initial underlying asset price
Q = 10
t= np.arange(1/365, 1, 2/12)

NUM_COLORS = len(t)
cm = plt.get_cmap('RdYlBu')
fig1 = plt.figure(1)
ax = fig1.add_subplot(111)

for i in range(len(t)):
    t_tmp = t[i]
    delta_call = cash_or_nothing_delta(spot, vol, r, q, t_tmp, K, Q, PutCall =  1)
    lines = ax.plot(spot, delta_call,label='delta')
    lines[0].set_color(cm(i/NUM_COLORS))

plt.xlabel("Asset price, S (USD)")
plt.ylabel("Cash-or-nothing call option delta (USD)")
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.grid(linestyle='--', axis='both', linewidth=0.25, color=[0.5,0.5,0.5])
plt.gca().legend(['T=1/365 Yr','T=2/12 Yr','T=4/12 Yr','T=6/12 Yr','T=8/12 Yr','T=10/12 Yr'],loc='upper left')

fig2 = plt.figure(2)
ax = fig2.add_subplot(111)
for i in range(len(t)):
    t_tmp = t[i]
    delta_put = cash_or_nothing_delta(spot, vol, r, q, t_tmp, K, Q, PutCall =  -1)
    lines = ax.plot(spot, delta_put, label='delta')
    lines[0].set_color(cm(i/NUM_COLORS))

plt.xlabel("Asset price, S (USD)")
plt.ylabel("Cash-or-nothing put option delta (USD)")
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.grid(linestyle='--', axis='both', linewidth=0.25, color=[0.5,0.5,0.5])
plt.gca().legend(['T=1/365 Yr','T=2/12 Yr','T=4/12 Yr','T=6/12 Yr','T=8/12 Yr','T=10/12 Yr'],loc='lower left')