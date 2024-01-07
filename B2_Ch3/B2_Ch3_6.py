# B2_Ch3_6.py

###############
# Prepared by Ran An, Wei Lu, and Feng Zhang
# Editor-in-chief: Weisheng Jiang, and Sheng Tu
# Book 2  |  Financial Risk Management with Python
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2021
###############

B2_Ch3_6_A.py 
import numpy as np
import pandas as pd
import pandas_datareader
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy.stats import norm
import random
import yfinance as yf
yf.pdr_override()

mpl.style.use('ggplot')
# extract stock data
ticker = 'AAPL'
stock = pd.DataFrame()
stock[ticker] = pandas_datareader.data.get_data_yahoo(ticker, start='2010-1-1', end='2019-12-31')['Adj Close']

stock.plot(figsize=(15,8), legend=None, c='r')
plt.title('Stock price for AAPL')
plt.xlabel('Time')
plt.ylabel('Stock price')


B2_Ch3_6_B.py 
# logarithmic returns
log_returns = np.log(1 + stock.pct_change())
ax = sns.distplot(log_returns.iloc[1:])
ax.set_xlabel("Daily Log Return")
ax.set_ylabel("Frequency")
ax.set_yticks([10, 20, 30, 40])


B2_Ch3_6_C.py 
# drift and volatility
u = log_returns.mean()
var = log_returns.var()
drift = u - (0.5*var)
stdev = log_returns.std()
print('Model mean: %.3f' % u)
print('Model variance: %.4f' % var)
print('Model drift: %.3f' % drift)
print('Model volatility: %.3f' % stdev)



B2_Ch3_6_D.py 
# daily returns and simulations
days = 60
MC_trials = 2000
random.seed(66)
Z = norm.ppf(np.random.rand(days, MC_trials)) 
daily_returns = np.exp(drift.values + stdev.values * Z)

price_paths = np.zeros_like(daily_returns)
price_paths[0] = stock.iloc[-1]
for t in range(1, days):
    price_paths[t] = price_paths[t-1]*daily_returns[t]

# plot paths and distribution for last day
rows = 1
cols = 2
fig, (ax1, ax2) = plt.subplots(rows, cols, figsize=(14,5), gridspec_kw={'width_ratios': [3, 1]})    
ax1.plot(price_paths, lw=0.5)
ax1.set_yticks([40, 70, 100, 130])
ax1.set_title('(a)', loc='left')
ax2 = sns.distplot(price_paths[-1], rug=True, rug_kws={"color": "green", "alpha": 0.5, "height": 0.06, "lw": 0.5}, vertical=True, label='(b)') # 

# ax2 = sns.distplot(price_paths[-1], rug=True, rug_kws={"color": "green", "alpha": 0.5, "height": 0.06, "lw": 0.5}, vertical=True, label='(b)')

# UserWarning: 

# `distplot` is a deprecated function and will be removed in seaborn v0.14.0.

# Please adapt your code to use either `displot` (a figure-level function with
# similar flexibility) or `histplot` (an axes-level function for histograms).

# For a guide to updating your code to use the new functions, please see
# https://gist.github.com/mwaskom/de44147ed2974457ad6372750bbe5751

# 但不知道如何处理 vertical 参数。

ax2.set_yticks([40, 70, 100, 130])
ax2.set_title('(b)', loc='left')

