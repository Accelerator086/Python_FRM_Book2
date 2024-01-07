# B2_Ch6_1.py

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
r = 0.03  #risk-free interest rate
q = 0.0   # dividend yield
vol = 0.5 #volatility
t_base = 2.0
PutCall = 1 # 1 for call;-1 for put
spot = 50
K = np.arange(20,80,1)  #strike price

plt.figure(1)
bs_price_call = option_analytical(spot, vol, r, q, t_base, K, PutCall =  1)
plt.plot(K, bs_price_call, label='price')
plt.xlabel("Strike price, K (USD)")
plt.ylabel("Euro call option price, C (USD)")
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.grid(linestyle='--', axis='both', linewidth=0.25, color=[0.5,0.5,0.5])
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

plt.figure(2)
bs_price_put = option_analytical(spot, vol, r, q, t_base, K, PutCall =  -1)
plt.plot(K, bs_price_put, label='price')
plt.xlabel("Strike price, K (USD)")
plt.ylabel("Euro put option price, P (USD)")
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.grid(linestyle='--', axis='both', linewidth=0.25, color=[0.5,0.5,0.5])