# B2_Ch1_3.py

###############
# Prepared by Ran An, Wei Lu, and Feng Zhang
# Editor-in-chief: Weisheng Jiang, and Sheng Tu
# Book 2  |  Financial Risk Management with Python
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2021
###############

import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader
import yfinance as yf
yf.pdr_override()
 
# sp500 price
sp500 = pandas_datareader.data.get_data_fred(['sp500'], start='12-28-2019', end='12-28-2020')

# daily log return
log_return_daily = np.log(sp500 / sp500.shift(1))
log_return_daily.dropna(inplace=True)
     
# calculate daily standard deviation of returns
daily_std = np.std(log_return_daily)[0]
 
# annualize daily standard deviation
std = daily_std * 252 ** 0.5
 
# Plot histograms
mpl.style.use('ggplot')
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
n, bins, patches = ax.hist(
    log_return_daily['sp500'],
    bins='auto', alpha=0.7, color='dodgerblue', rwidth=0.85)

ax.set_xlabel('Log return')
ax.set_ylabel('Frequency of log return')
ax.set_title('Historical volatility for SP500')
 
# get x and y coordinate limits
x_corr = ax.get_xlim()
y_corr = ax.get_ylim()
 
# make room for text
header = y_corr[1] / 5
y_corr = (y_corr[0], y_corr[1] + header)
ax.set_ylim(y_corr[0], y_corr[1])

# print historical volatility on plot
x = x_corr[0] + (x_corr[1] - x_corr[0]) / 30
y = y_corr[1] - (y_corr[1] - y_corr[0]) / 15
ax.text(x, y , 'Annualized volatility: ' + str(np.round(std*100, 1))+'%',
    fontsize=11, fontweight='bold')
x = x_corr[0] + (x_corr[1] - x_corr[0]) / 15
y -= (y_corr[1] - y_corr[0]) / 20

fig.tight_layout()