# B2_Ch6_2.py

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

def option_analytical(S0, vol, r, q, t, K, PutCall):
    d1 = (np.log(S0 / K) + (r - q + 0.5 * vol ** 2) * t) / (vol * np.sqrt(t))
    d2 = (np.log(S0 / K) + (r - q - 0.5 * vol ** 2) * t) / (vol * np.sqrt(t))
    price =  PutCall*S0 * np.exp(-q * t) * norm.cdf(PutCall*d1, 0.0, 1.0) - PutCall* K * np.exp(-r * t) * norm.cdf(PutCall*d2, 0.0, 1.0)  
        
    return price

    # Inputs
r = 0.085  #risk-free interest rate
q = 0.0 # dividend yield
K = 55   #strike price
vol = 0.45 #volatility
t_base = 2.0
PutCall = 1 # 1 for call;-1 for put
spot = np.arange(0,100,1)  #initial underlying asset price
t= np.arange(0.00001, 2, 0.25)

NUM_COLORS = len(t)
cm = plt.get_cmap('bwr')
cm = plt.get_cmap('RdYlBu')
fig1 = plt.figure(1)
ax = fig1.add_subplot(111)

for i in range(len(t)):
    t_tmp = t[i]
    bs_price_call = option_analytical(spot, vol, r, q, t_tmp, K, PutCall =  1)
    lines = ax.plot(spot, bs_price_call, label='price')
    lines[0].set_color(cm(i/NUM_COLORS))

plt.xlabel("Stock price, S (USD)")
plt.ylabel("Euro call option price, C (USD)")
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.grid(linestyle='--', axis='both', linewidth=0.25, color=[0.5,0.5,0.5])
plt.gca().legend(['T = 0.00 Yr','T = 0.25 Yr','T = 0.50 Yr','T = 0.75 Yr','T = 1.00 Yr','T = 1.25 Yr','T = 1.50 Yr','T = 1.75 Yr'],loc='upper left')

fig2 = plt.figure(2)
ax = fig2.add_subplot(111)
for i in range(len(t)):
    t_tmp = t[i]
    bs_price_put = option_analytical(spot, vol, r, q, t_tmp, K, PutCall =  -1)
    lines = ax.plot(spot, bs_price_put, label='price')
    lines[0].set_color(cm(i/NUM_COLORS))

plt.xlabel("Stock price, S (USD)")
plt.ylabel("Euro put option price, P (USD)")
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.grid(linestyle='--', axis='both', linewidth=0.25, color=[0.5,0.5,0.5])
plt.gca().legend(['T = 0.00 Yr','T = 0.25 Yr','T = 0.50 Yr','T = 0.75 Yr','T = 1.00 Yr','T = 1.25 Yr','T = 1.50 Yr','T = 1.75 Yr'],loc='upper right')
