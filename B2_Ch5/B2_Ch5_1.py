# B2_Ch5_1.py

###############
# Prepared by Ran An, Wei Lu, and Feng Zhang
# Editor-in-chief: Weisheng Jiang, and Sheng Tu
# Book 2  |  Financial Risk Management with Python
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2021
###############

import numpy as np
import matplotlib.pyplot as plt

def generic_payoff(buy_or_sell, put_call_indicator, strike, spot):
    payoff = buy_or_sell*np.maximum(put_call_indicator*(spot - strike),0)
    return payoff

def generic_pnl(buy_or_sell, put_call_indicator, strike, spot, premium):
    pnl = buy_or_sell*np.maximum(put_call_indicator*(spot - strike),0) - buy_or_sell*premium
    return pnl

def plot_decor(x, y, y_label):
    plt.figure()
    plt.plot(x, y)
    plt.xlabel('S, spot price of underlying at expiration', fontsize=8)
    plt.ylabel(y_label,fontsize=8) 
    plt.gca().set_aspect(1)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['left'].set_position(('data',0))
    plt.gca().spines['bottom'].set_position(('data',0))

    plt.axvline(x=strike, linestyle='--', color='r', linewidth = .5)

    plt.xticks(np.arange(0, 200, step=20))
    plt.yticks(np.arange(np.floor(np.min(pay_off)/10.0)*10.0, np.ceil(np.max(pay_off)/10.0)*10.0, step=20))
    plt.grid(linestyle='--', axis='both', linewidth=0.25, color=[0.5,0.5,0.5])

put_call_indicator = 1; 
# 1 for call; -1 for put
buy_or_sell = 1;
# 1 for buy ;-1 for sell
strike = 100 
spot = np.arange(0,200,1)
premium = 10 

# long a call

pay_off = generic_payoff(buy_or_sell,put_call_indicator,strike,spot)
y_label = 'Payoff';
plot_decor(spot, pay_off, y_label)

pnl = generic_pnl(buy_or_sell,put_call_indicator,strike,spot,premium)
y_label = 'PnL';
plot_decor(spot, pnl, y_label)

# short a call

put_call_indicator = 1; 
buy_or_sell = -1;
pay_off = generic_payoff(buy_or_sell,put_call_indicator,strike,spot)
y_label = 'Pay off';
plot_decor(spot, pay_off, y_label)

pnl = generic_pnl(buy_or_sell,put_call_indicator,strike,spot,premium)
y_label = 'PnL';
plot_decor(spot, pnl, y_label)

# long a put

put_call_indicator = -1; 
buy_or_sell = 1;
pay_off = generic_payoff(buy_or_sell,put_call_indicator,strike,spot)
y_label = 'Pay off';
plot_decor(spot, pay_off, y_label)

pnl = generic_pnl(buy_or_sell,put_call_indicator,strike,spot,premium)
y_label = 'PnL';
plot_decor(spot, pnl, y_label)

# short a put

put_call_indicator = -1; 
buy_or_sell = -1;
pay_off = generic_payoff(buy_or_sell,put_call_indicator,strike,spot)
y_label = 'Pay off';
plot_decor(spot, pay_off, y_label)

pnl = generic_pnl(buy_or_sell,put_call_indicator,strike,spot,premium)
y_label = 'PnL';
plot_decor(spot, pnl, y_label)
