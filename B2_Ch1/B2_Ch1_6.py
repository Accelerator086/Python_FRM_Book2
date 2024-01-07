# B2_Ch1_6.py

###############
# Prepared by Ran An, Wei Lu, and Feng Zhang
# Editor-in-chief: Weisheng Jiang, and Sheng Tu
# Book 2  |  Financial Risk Management with Python
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2021
###############

import pandas_datareader
import matplotlib.pyplot as plt 
import numpy as np
import yfinance as yf
yf.pdr_override()

# sp500 price
df = pandas_datareader.data.get_data_fred(['sp500'], start='12-28-2010', end='12-28-2020')
df.dropna(inplace=True)

# daily log return
df['Daily return squared'] = np.log(df['sp500'] / df['sp500'].shift(1))*np.log(df['sp500'] / df['sp500'].shift(1))
df.dropna(inplace=True)

# calculate exponentially weighted moving average
alpha_list = [0.01, 0.03, 0.06]
for alpha in alpha_list:
    ma = df['sp500'].ewm(alpha=alpha, adjust=False).std()
    df[alpha] = ma
    df.rename(columns={alpha:'$\lambda$ = '+str(1-alpha)}, inplace=True)

# plot dataframe
# sp500 price
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
ax1.plot(df['sp500'])
ax1.set_title('SP500 price')
ax1.set_xlabel("Date")
ax1.set_ylabel("Price")
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.yaxis.set_ticks_position('left')
ax1.xaxis.set_ticks_position('bottom')

# daily log return squared
ax2.plot(df['Daily return squared'])
ax2.set_title('Daily return squared')
ax2.set_xlabel("Date")
ax2.set_ylabel("Daily return squared")
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.yaxis.set_ticks_position('left')
ax2.xaxis.set_ticks_position('bottom')
# ewma vol
ax3.plot(df.loc[:, (df.columns != 'sp500') & (df.columns != 'Daily return squared')])
ax3.legend(df.loc[:, (df.columns != 'sp500') & (df.columns != 'Daily return squared')].columns)
ax3.set_title('SP500 price volatility via EWMA analysis')
ax3.set_xlabel("Date")
ax3.set_ylabel("Volatility")
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax3.yaxis.set_ticks_position('left')
ax3.xaxis.set_ticks_position('bottom')

fig.tight_layout()