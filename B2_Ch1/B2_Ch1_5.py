# B2_Ch1_5.py

###############
# Prepared by Ran An, Wei Lu, and Feng Zhang
# Editor-in-chief: Weisheng Jiang, and Sheng Tu
# Book 2  |  Financial Risk Management with Python
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2021
###############

import pandas_datareader
import matplotlib.pyplot as plt 
import yfinance as yf
yf.pdr_override()

# sp500 price
df = pandas_datareader.data.get_data_fred(['sp500'], start='12-28-2010', end='12-28-2020')
df.dropna(inplace=True)

# calculate cumulative moving average
df['cma'] = df['sp500'].expanding(1).std()
df.dropna(inplace=True)
# df.rename(columns={win:'Vol via '+str(win)+' days MA'}, inplace=True)

# plot dataframe
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
ax1.plot(df['sp500'])
ax1.set_title('SP500 price')
ax1.set_xlabel("Date")
ax1.set_ylabel("Price")
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.yaxis.set_ticks_position('left')
ax1.xaxis.set_ticks_position('bottom')

ax2.plot(df['cma'])
ax2.set_title('SP500 price volatility via cumulative moving average analysis')
ax2.set_xlabel("Date")
ax2.set_ylabel("Volatility")
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.yaxis.set_ticks_position('left')
ax2.xaxis.set_ticks_position('bottom')

fig.tight_layout()
