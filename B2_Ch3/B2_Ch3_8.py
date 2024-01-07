# B2_Ch3_8.py

###############
# Prepared by Ran An, Wei Lu, and Feng Zhang
# Editor-in-chief: Weisheng Jiang, and Sheng Tu
# Book 2  |  Financial Risk Management with Python
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2021
###############

B2_Ch3_8_A.py 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime

# underlying price
S0 = 857.29 
# volatility
v = 0.2076 
# risk free interest rate
r = 0.0014 # rate of 0.14%
# maturity
T = (datetime.date(2020,9,30) - datetime.date(2020,9,1)).days / 252.0
# strike price
K = 900.
# monte carlo simulation numbers
MC_num = 1000

ST_list = []
payoff_list = []
discount_factor = np.exp(-r * T)
# monte carlo simulation
for i in range(MC_num):
    ST = S0 * np.exp((r - 0.5 * v**2) * T + v * np.sqrt(T) * np.random.normal(0,1.0))
    ST_list.append(ST)
    payoff = max(0.0, ST-K)
    payoff_list.append(payoff)

# plot simulated asset price
mpl.style.use('ggplot')
plt.plot(ST_list, 'o', color='#3C9DFF', markersize=5)
plt.hlines(S0, 0, MC_num, colors='g', linestyles='--',label='Initial asset price')
plt.text(MC_num+1, S0, 'Initial asset price')
plt.hlines(K, 0, MC_num, colors='r', linestyles='--',label='Strike price')
plt.text(MC_num+1, K, 'Strike price')
plt.title("Monte Carlo simulation for asset price")
plt.xlabel("Number of simulations")
plt.ylabel("Simulated asset price")



B2_Ch3_8_B.py 
option_price = discount_factor * (sum(payoff_list) / float(MC_num))
print ('European call option price: %.2f' % option_price)
