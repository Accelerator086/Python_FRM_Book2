# B2_Ch2_1.py

###############
# Prepared by Ran An, Wei Lu, and Feng Zhang
# Editor-in-chief: Weisheng Jiang, and Sheng Tu
# Book 2  |  Financial Risk Management with Python
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2021
###############

from matplotlib import pyplot as plt
import numpy as np
# import pandas_datareader
import matplotlib as mpl
mpl.style.use('ggplot')
# import yfinance as yf
# yf.pdr_override()

## real stock price
# stock: Twitter Inc
ticker = 'TWTR'

# calibration period
start_date = '2018-10-01'
end_date = '2020-10-01'

# extract and plot historical stock data
# stock = pandas_datareader.data.get_data_quandl(ticker, start=start_date, end=end_date)['Adj Close']

# akshare?
import akshare as ak

stock_us_daily_df = ak.stock_us_daily(symbol=ticker, adjust="")
stock_us_daily_df = stock_us_daily_df[(stock_us_daily_df['date'] >= start_date) & (stock_us_daily_df['date'] <= end_date)]
# print(stock_us_daily_df)
stock = stock_us_daily_df['close']

## simulated stock price
np.random.seed(66)
def gbm(S,v,r,T):
    return S * np.exp((r - 0.5 * v**2) * T + v * np.sqrt(T) * np.random.normal(0,1.0))

# initial
S0 = 26.68 
# volatility
vol = 0.8865
# mu
mu = 0.35 
# time increment
dt = 1/252
# maturity in year
T = 2 
# step numbers
N = int(T/dt) 

path=[]
S=S0
for i in range(1,N+1):
    S_t = gbm(S,vol,mu,dt)
    S= S_t
    path.append(S_t)

## plot stock price
rows = 2
cols = 1
fig, (ax1, ax2) = plt.subplots(rows, cols, figsize=(14,8))
# real stock price
ax1.plot(stock)
ax1.set_title('(a) Stock price for TWTR', loc='left')
ax1.set_xlabel('Time')
ax1.set_ylabel('Real Stock price')
# simulated stochastic process
ax2.plot(path)
ax2.set_title('(b) Simulated stochastic process', loc='left')
ax2.set_xlabel('t')
ax2.set_ylabel('S')
plt.tight_layout()
