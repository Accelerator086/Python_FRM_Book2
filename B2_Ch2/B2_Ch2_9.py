# B2_Ch2_9.py

###############
# Prepared by Ran An, Wei Lu, and Feng Zhang
# Editor-in-chief: Weisheng Jiang, and Sheng Tu
# Book 2  |  Financial Risk Management with Python
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2021
###############

B2_Ch2_9_A.py
import pandas as pd
import numpy as np
import pandas_datareader
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('ggplot')
import yfinance as yf
yf.pdr_override()

# stock: Apple Inc.
ticker = 'AAPL'

# calibration period
start_date = '2010-9-1'
end_date = '2020-9-30' 

# extract and plot historical stock data
stock = pandas_datareader.data.get_data_yahoo(ticker, start=start_date, end=end_date)['Adj Close']
stock.plot(figsize=(15,8), legend=None, c='r')
plt.title('Stock price for AAPL')
plt.xlabel('Time')
plt.ylabel('Stock price')


B2_Ch2_9_B.py
# stock log returns
log_returns = np.log(1 + stock.pct_change())

# inital stock price
S0 = stock.iloc[-1]
# time increment
dt = 1 
# end date of prediction
pred_end_date = '2020-10-31' 
# days of prediction time horizon
T = pd.date_range(start = pd.to_datetime(end_date, format = "%Y-%m-%d") + pd.Timedelta('1 days'), 
                 end = pd.to_datetime(pred_end_date,format = "%Y-%m-%d")).to_series().map(lambda x: 1 if x.isoweekday() in range(1,6) else 0).sum()

# simulation steps
N = int(T / dt)
# mean
mu = np.mean(log_returns)
print('Model mean: %.3f' % mu)
# volitality
vol = np.std(log_returns)
print('Model volatility: %.3f' % vol)


B2_Ch2_9_C.py
S =  [None] * (N+2)
S[0] = S0
for t in range(1, N+2):
    # calculate drift and diffusion
    drift = (mu - 0.5 * vol**2) * dt
    diffusion = vol * np.random.normal(0, 1.0)  
    # predict stock price
    daily_returns = np.exp(drift + diffusion)
    S[t] = S[t-1]*daily_returns

# plot simulations
plt.figure(figsize = (15,8))
plt.title("Stock price prediction ")
plt.plot(pd.date_range(start = stock.index.max(), 
                       end = pred_end_date, freq = 'D').map(lambda x: x if x.isoweekday() in range(1, 6) else np.nan).dropna(), 
         S)
plt.xlabel('Time')
plt.xticks(rotation = 45, ha='center')
plt.ylabel('Stock price')
