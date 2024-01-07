# B2_Ch6_5.py

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

def cash_or_nothing_analytical(S0, vol, r, q, t, K, Q, PutCall):
    if t == 0:
        price =  Q*np.array(PutCall*(S0-K)>=0,dtype =bool)  
    elif t > 0:
        d2 = (np.log(S0 / K) + (r - q - 0.5 * vol ** 2) * t) / (vol * np.sqrt(t))
       
        price =  Q*np.exp(-r * t) * norm.cdf(PutCall*d2, 0.0, 1.0)      
    else:
        print("time to maturity should be greater or equal to zero")
    return price

    # Inputs
r = 0.085  #risk-free interest rate
q = 0.0 # dividend yield
K = 55   #strike price
vol = 0.45 #volatility
PutCall = 1 # 1 for call;-1 for put
spot = np.arange(10,105,1)  #initial underlying asset price
Q = 1
t = 0

plt.figure(1)
price_call = cash_or_nothing_analytical(spot, vol, r, q, t, K, Q, PutCall =  1)
plt.plot(spot, price_call, '.',label='price')

plt.xlabel("Stock price, S (USD)")
plt.ylabel("Payoff at expiration")
plt.grid(linestyle='--', axis='both', linewidth=0.25, color=[0.5,0.5,0.5])
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

plt.figure(2)
price_put = cash_or_nothing_analytical(spot, vol, r, q, t, K, Q, PutCall =  -1)
plt.plot(spot, price_put, '.',label='price')

plt.xlabel("Stock price, S (USD)")
plt.ylabel("Payoff at expiration")
plt.grid(linestyle='--', axis='both', linewidth=0.25, color=[0.5,0.5,0.5])
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)