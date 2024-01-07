# B2_Ch1_9.py

###############
# Prepared by Ran An, Wei Lu, and Feng Zhang
# Editor-in-chief: Weisheng Jiang, and Sheng Tu
# Book 2  |  Financial Risk Management with Python
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2021
###############

import numpy as np
from scipy import stats

# function to calculate price of options (call or put) by BS
def option_price_BS(option_type, sigma, s, k, r, T, q=0.0):    
    d1 = (np.log(s / k) + (r - q + sigma ** 2 * 0.5) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        option_price = np.exp(-r*T) * (s * np.exp((r - q)*T) * stats.norm.cdf(d1) - k *  stats.norm.cdf(d2))
        return option_price
    elif option_type == 'put':
        option_price = np.exp(-r*T) * (k * stats.norm.cdf(-d2) - s * np.exp((r - q)*T) *  stats.norm.cdf(-d1))
        return option_price
    else:
        print('Option type should be call or put.')

# funciton to calculate implied volatility by bisection method
def implied_vol(option_type, option_price, s, k, r, T, q):  
    precision = 0.00001
    upper_vol = 500.0
    lower_vol = 0.0001
    iteration = 0

    while iteration >= 0:

        iteration +=1
        mid_vol = (upper_vol + lower_vol)/2.0
        price = option_price_BS(option_type, mid_vol, s, k, r, T, q)
        
        if option_type == 'call':
            lower_price = option_price_BS(option_type, lower_vol, s, k, r, T, q)
            if (lower_price - option_price) * (price - option_price) > 0:
                lower_vol = mid_vol
            else:
                upper_vol = mid_vol
            if abs(price - option_price) < precision: 
                break 

        elif option_type == 'put':
            upper_price = option_price_BS(option_type, upper_vol, s, k, r, T, q)

            if (upper_price - option_price) * (price - option_price) > 0:
                upper_vol = mid_vol
            else:
                lower_vol = mid_vol
            if abs(price - option_price) < precision: 
                break 
            
        if iteration == 1000: 
            break
    print('Implied volatility: %.4f' % mid_vol)
    return mid_vol


implied_vol('call', 17.5, 586.08, 585, 0.0002, 30.0/365, 0.0)
