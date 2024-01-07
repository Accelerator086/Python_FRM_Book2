# B2_Ch6_9.py

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

def option_delta(S0, vol, r, q, t, K, PutCall):
    d1 = (np.log(S0 / K) + (r - q + 0.5 * vol ** 2) * t) / (vol * np.sqrt(t))
    if PutCall == 1:
        delta = np.exp(-q * t) * norm.cdf(d1, 0.0, 1.0)
    else:    
        delta = np.exp(-q * t) * (norm.cdf(d1, 0.0, 1.0) -1 )
        
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

EPSILO = 0.05*K
N = 0.5*Q/EPSILO 
NUM_COLORS = len(t)
cm = plt.get_cmap('RdYlBu')
fig1 = plt.figure(1)
ax = fig1.add_subplot(111)

for i in range(len(t)):
    t_tmp = t[i]
    delta_call_K1 = option_delta(spot, vol, r, q, t_tmp, K - EPSILO, PutCall =  1)
    delta_call_K2 = option_delta(spot, vol, r, q, t_tmp, K + EPSILO, PutCall =  1)
    lines = ax.plot(spot, N*(delta_call_K1 - delta_call_K2)  ,label='delta')
    lines[0].set_color(cm(i/NUM_COLORS))

plt.xlabel("Asset price, S (USD)")
plt.ylabel("replicating Cash-or-nothing call option delta (USD)")
plt.grid(linestyle='--', axis='both', linewidth=0.25, color=[0.5,0.5,0.5])
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().legend(['T=1/365 Yr','T=2/12 Yr','T=4/12 Yr','T=6/12 Yr','T=8/12 Yr','T=10/12 Yr'],loc='upper left')

fig2 = plt.figure(2)
ax = fig2.add_subplot(111)

for i in range(len(t)):
    t_tmp = t[i]
    delta_put_K1 = option_delta(spot, vol, r, q, t_tmp, K + EPSILO, PutCall =  -1)
    delta_put_K2 = option_delta(spot, vol, r, q, t_tmp, K - EPSILO, PutCall =  -1)
    lines = ax.plot(spot, N*(delta_put_K1 - delta_put_K2), label='delta')
    lines[0].set_color(cm(i/NUM_COLORS))

plt.xlabel("Asset price, S (USD)")
plt.ylabel("replicating Cash-or-nothing put option delta (USD)")
plt.grid(linestyle='--', axis='both', linewidth=0.25, color=[0.5,0.5,0.5])
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().legend(['T=1/365 Yr','T=2/12 Yr','T=4/12 Yr','T=6/12 Yr','T=8/12 Yr','T=10/12 Yr'],loc='lower left')
