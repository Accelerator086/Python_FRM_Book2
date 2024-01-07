# B2_Ch6_10.py

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

def asset_or_nothing_analytical(S0, vol, r, q, t, K, PutCall):
    if t == 0:
        price =  S0*np.array(PutCall*(S0-K)>=0,dtype =bool)  
    elif t > 0:
        d1 = (np.log(S0 / K) + (r - q + 0.5 * vol ** 2) * t) / (vol * np.sqrt(t))       
        price =  S0*np.exp(-q * t) * norm.cdf(PutCall*d1, 0.0, 1.0)      
    else:
        print("time to maturity should be greater or equal to zero")
    return price

    # Inputs
r = 0.085  #risk-free interest rate
q = 0.0 # dividend yield
K = 55   #strike price
vol = 0.45 #volatility
PutCall = 1 # 1 for call;-1 for put
spot = np.arange(10,105,0.2)  #initial underlying asset price
Q = K
t = 0

plt.figure(1)
asset_or_nothing_call = asset_or_nothing_analytical(spot, vol, r, q, t, K, PutCall =  1)
plt.plot(spot, asset_or_nothing_call, '.',label='price')

plt.xlabel("Stock price, S (USD)")
plt.ylabel("Payoff at expiration")
plt.grid(linestyle='--', axis='both', linewidth=0.25, color=[0.5,0.5,0.5])
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

plt.figure(2)
asset_or_nothing_put = asset_or_nothing_analytical(spot, vol, r, q, t, K, PutCall =  -1)
plt.plot(spot, asset_or_nothing_put, '.',label='price')

plt.xlabel("Stock price, S (USD)")
plt.ylabel("Payoff at expiration")
plt.grid(linestyle='--', axis='both', linewidth=0.25, color=[0.5,0.5,0.5])
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

