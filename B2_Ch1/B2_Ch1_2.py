# B2_Ch1_2.py

###############
# Prepared by Ran An, Wei Lu, and Feng Zhang
# Editor-in-chief: Weisheng Jiang, and Sheng Tu
# Book 2  |  Financial Risk Management with Python
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2021
###############

B2_Ch1_2_A.py
import pandas_datareader 
import yfinance as yf
yf.pdr_override()
import matplotlib.pyplot as plt

# sp500 price
sp500 = pandas_datareader.data.get_data_fred(['sp500'], start='2010-12-28', end='2020-12-28')
# plot sp500 price
plt.plot(sp500['sp500'], color='dodgerblue')
plt.title('S&P 500 price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().yaxis.set_ticks_position('left')
plt.gca().xaxis.set_ticks_position('bottom')

B2_Ch1_2_B.py
# daily return
sp500['return_daily'] = sp500['sp500'].pct_change()
sp500.dropna(inplace=True)
# plot daily return
plt.plot(sp500['return_daily'], color='dodgerblue')
plt.title('S&P 500 daily returns')
plt.xlabel('Date')
plt.ylabel('Daily return')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().yaxis.set_ticks_position('left')
plt.gca().xaxis.set_ticks_position('bottom')

B2_Ch1_2_C.py
# monthly return
sp500_monthly_returns = sp500['sp500'].resample('M').ffill().pct_change() #周、双周回报率：W, BW
# plot monthly return
plt.plot(sp500_monthly_returns, color='dodgerblue')
plt.title('S&P 500 monthly returns')
plt.xlabel('Date')
plt.ylabel('Monthly return')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().yaxis.set_ticks_position('left')
plt.gca().xaxis.set_ticks_position('bottom')

B2_Ch1_2_D.py
# daily cumulative return
sp500_cum_returns_daily = (sp500['return_daily'] + 1).cumprod()
# plot daily cumulative return
plt.plot(sp500_cum_returns_daily, color='dodgerblue')
plt.title('S&P 500 daily cumulative returns')
plt.xlabel('Date')
plt.ylabel('Cumulative return')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().yaxis.set_ticks_position('left')
plt.gca().xaxis.set_ticks_position('bottom')

B2_Ch1_2_E.py
# monthly cumulative return
sp500_cum_returns_monthly = (sp500_monthly_returns + 1).cumprod()
# plot monthly cumulative return
plt.plot(sp500_cum_returns_monthly, color='dodgerblue')
plt.title('S&P 500 daily cumulative returns')
plt.xlabel('Date')
plt.ylabel('Cumulative return')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().yaxis.set_ticks_position('left')
plt.gca().xaxis.set_ticks_position('bottom')