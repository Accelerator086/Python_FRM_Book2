# B2_Ch6_6.py

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
PutCall = 1 # 1 for call;-1 for put
spot = np.arange(10,105,0.2)  #initial underlying asset price
Q = 1
t = 0
EPSILO = 0.01*K
N = 0.5*Q/EPSILO 
EPSILON = np.arange(0.005,0.2,0.025)*K

NUM_COLORS = len(EPSILON )
cm = plt.get_cmap('RdYlBu')
fig1 = plt.figure(1)
ax = fig1.add_subplot(111)

for i in range(len(EPSILON)):
    EPSILON_tmp = EPSILON[i]
    european_call_K1 = option_analytical(spot, vol, r, q, t, K-EPSILON_tmp, PutCall =  1)
    european_call_K2 = option_analytical(spot, vol, r, q, t, K+EPSILON_tmp, PutCall =  1)
    N_tmp = 0.5*Q/EPSILON_tmp 
    lines = ax.plot(spot, N_tmp*european_call_K1-N_tmp*european_call_K2,label='price')
    lines[0].set_color(cm(i/NUM_COLORS))
plt.xlabel("Stock price, S (USD)")
plt.ylabel("Payoff at expiration")
plt.grid(linestyle='--', axis='both', linewidth=0.25, color=[0.5,0.5,0.5])
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().legend(['e = 0.275','e = 1.65','e = 3.025','e = 4.4','e = 5.775','e = 7.15','e = 8.525','e = 9.9'],loc='upper left')

fig2 = plt.figure(2)
ax = fig2.add_subplot(111)

for i in range(len(EPSILON)):
    EPSILON_tmp = EPSILON[i]
    european_put_K1 = option_analytical(spot, vol, r, q, t, K+EPSILON_tmp, PutCall =  -1)
    european_put_K2 = option_analytical(spot, vol, r, q, t, K-EPSILON_tmp, PutCall =  -1)
    N_tmp = 0.5*Q/EPSILON_tmp 
    lines = ax.plot(spot, N_tmp*european_put_K1-N_tmp*european_put_K2,label='price')
    lines[0].set_color(cm(i/NUM_COLORS))
plt.xlabel("Stock price, S (USD)")
plt.ylabel("Payoff at expiration")
plt.grid(linestyle='--', axis='both', linewidth=0.25, color=[0.5,0.5,0.5])
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().legend(['e = 0.275','e = 1.65','e = 3.025','e = 4.4','e = 5.775','e = 7.15','e = 8.525','e = 9.9'],loc='upper right')