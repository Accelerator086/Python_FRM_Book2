# B2_Ch6_3.py

###############
# Prepared by Ran An, Wei Lu, and Feng Zhang
# Editor-in-chief: Weisheng Jiang, and Sheng Tu
# Book 2  |  Financial Risk Management with Python
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2021
###############

import numpy as np
from scipy.stats import norm

def option_analytical(S0, vol, r_d, r_f, t, K, PutCall):
    d1 = (np.log(S0 / K) + (r_d - r_f + 0.5 * vol ** 2) * t) / (vol * np.sqrt(t))
    d2 = (np.log(S0 / K) + (r_d - r_f - 0.5 * vol ** 2) * t) / (vol * np.sqrt(t))  
    price =  PutCall*S0 * np.exp(-r_f * t) * norm.cdf(PutCall*d1, 0.0, 1.0) - PutCall* K * np.exp(-r_d * t) * norm.cdf(PutCall*d2, 0.0, 1.0)  
        
    return price

    # Inputs
S0 = 1.6   # spot price, units of domestic currency of one unit of foreign currency
r_d = 0.0606  # domestic risk-free interest rate
r_f = 0.1168   # forieign risk-free interest rate 
K = 1.58    # strike price, units of domestic currency of one unit of foreign currency 
vol = 0.15    #volatility
t = 90/365
PutCall = -1 # 1 for call;-1 for put

bs_price = option_analytical(S0, vol, r_d, r_f, t, K, PutCall)
print('analytical Price: %.4f' % bs_price)