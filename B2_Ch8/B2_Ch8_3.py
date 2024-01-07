# B2_Ch8_3.py

###############
# Prepared by Ran An, Wei Lu, and Feng Zhang
# Editor-in-chief: Weisheng Jiang, and Sheng Tu
# Book 2  |  Financial Risk Management with Python
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2021
###############

B2_Ch8_3_A.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader
import scipy.stats as stats
from mpl_toolkits import mplot3d
from matplotlib import cm
import yfinance as yf
yf.pdr_override()

    
tickers = ['GOOGL','META','AAPL','NFLX','AMZN']
ticker_num = len(tickers)
price_data = []
for ticker in range(ticker_num):   
    # prices = pandas_datareader.DataReader(tickers[ticker], start='2015-11-30', end = '2020-11-30', data_source='yahoo') 
    prices = pandas_datareader.data.get_data_yahoo(tickers[ticker], start='2015-11-30', end = '2020-11-30')   
    price_data.append(prices[['Adj Close']])
    df_stocks = pd.concat(price_data, axis=1)
    
# stock log returns
logreturns = np.log(df_stocks/df_stocks.shift(1))[1:]  
logreturns.columns = tickers 
logreturns.head() 

B2_Ch8_3_B.py
# plot log return distribution for GOOGL
plt.style.use('ggplot')
mu, std = stats.norm.fit(logreturns['GOOGL'])
x = np.linspace(mu-5*std, mu+5*std, 500)
logreturns['GOOGL'].hist(bins=60, density=True, histtype="stepfilled", alpha=0.5)
x = np.linspace(mu - 3*std, mu+3*std, 500)
plt.plot(x, stats.norm.pdf(x, mu, std))
plt.title("Log return distribution for GOOGL")
plt.xlabel("Return")
plt.ylabel("Density")

B2_Ch8_3_C.py
# plot log return distribution
rows = 2
cols = 2
fig, axs = plt.subplots(rows, cols, figsize=(12,6))
ticker_n = 1
for i in range(rows):
    for j in range(cols):
        mu, std = stats.norm.fit(logreturns[tickers[ticker_n]])
        x = np.linspace(mu-5*std, mu+5*std, 500)
        axs[i,j].hist(logreturns[tickers[ticker_n]], bins=60, density=True, histtype="stepfilled", alpha=0.5)
        axs[i,j].plot(x, stats.norm.pdf(x, mu, std))
        axs[i,j].set_title("Log return distribution for "+tickers[ticker_n])
        axs[i,j].set_xlabel("Return")
        axs[i,j].set_ylabel("Density")
        ticker_n = ticker_n + 1
plt.tight_layout()


B2_Ch8_3_D.py
# covariance matrix
cov_logreturns = logreturns.cov() 
# mean returns for each stock
mean_logreturns = logreturns.mean() 
# weights for stocks in the portfolio
stock_weight = np.array([0.2, 0.3, 0.1, 0.15, 0.25]) 
# mean returns and volitality for portfolio  
portfolio_mean_log = mean_logreturns.dot(stock_weight) 
portfolio_vol_log = np.sqrt(np.dot(stock_weight.T, np.dot(cov_logreturns, stock_weight)))
print('The mean and volatility of the portfolio are {:.6f} and {:.6f}, respectively.'.format(portfolio_mean_log, portfolio_vol_log))


B2_Ch8_3_E.py
# confidence level
confidence_level = 0.99
# VaR calculation: initial investment value and holding period
initial_investment = 1000000
n = 1
VaR_norm = initial_investment*(portfolio_vol_log*abs(stats.norm.ppf(q=1-confidence_level))-portfolio_mean_log)*np.sqrt(n)
VaR_lognorm = initial_investment*(1-np.exp(portfolio_mean_log-portfolio_vol_log*abs(stats.norm.ppf(q=1-confidence_level))))*np.sqrt(n)
print('The normal VaR and lognormal VaR of the portfolio in 1 day holding period are {:.0f} and {:.0f}, respectively.'.format(VaR_norm, VaR_lognorm))



B2_Ch8_3_F.py
# confidence level list
confidence_level_list = np.arange(0.90, 0.99, 0.001)
# initial investment value
initial_investment = 1000000
n = 1
VaR_norm_list = []
VaR_lognorm_list = []
for confidence_level in confidence_level_list:
    VaR_norm = initial_investment*(portfolio_vol_log*abs(stats.norm.ppf(q=1-confidence_level))-portfolio_mean_log)*np.sqrt(n)
    VaR_norm_list.append(VaR_norm)
    VaR_lognorm = initial_investment*(1-np.exp(portfolio_mean_log-portfolio_vol_log*abs(stats.norm.ppf(q=1-confidence_level))))*np.sqrt(n)
    VaR_lognorm_list.append(VaR_lognorm)
plt.plot(confidence_level_list, VaR_norm_list, label='Normal VaR')
plt.plot(confidence_level_list, VaR_lognorm_list, label='Lognormal VaR')
plt.legend()
plt.xlabel('Confidence level')
plt.ylabel('1-day VaR')



B2_Ch8_3_G.py
# 3D display 
holding_period_list = np.arange(1,91,1)
fig = plt.figure()
ax = plt.axes(projection='3d')
xdata = confidence_level_list
ydata = holding_period_list
x3d, y3d = np.meshgrid(xdata, ydata)
z3d = initial_investment*(portfolio_vol_log*abs(stats.norm.ppf(q=1-x3d))-portfolio_mean_log)*np.sqrt(y3d)
ax.plot_wireframe(x3d, y3d, z3d, rstride=4, cstride=4, linewidth=1, color='black')
ax.plot_surface(x3d, y3d, z3d, rstride=4, cstride=4, alpha=0.4,cmap=plt.cm.summer)
ax.set_xlabel('\nConfidence level')
ax.set_ylabel('\nHolding period')
ax.set_zlabel('\nVaR')