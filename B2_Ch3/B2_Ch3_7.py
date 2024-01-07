# B2_Ch3_7.py

###############
# Prepared by Ran An, Wei Lu, and Feng Zhang
# Editor-in-chief: Weisheng Jiang, and Sheng Tu
# Book 2  |  Financial Risk Management with Python
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2021
###############

B2_Ch3_7_A.py 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader
import matplotlib as mpl
import yfinance as yf
yf.pdr_override()

tickers = ['AAPL','AMZN']
ticker_num = len(tickers)
price_data = []
for ticker in range(ticker_num):   
    prices = pandas_datareader.data.get_data_yahoo(tickers[ticker], start='2016-01-15', end = '2021-01-15')   
    price_data.append(prices[['Adj Close']])
    df_stocks = pd.concat(price_data, axis=1)
df_stocks.columns = tickers

mpl.style.use('ggplot')
fig, axs = plt.subplots(2, 1, figsize=(14,8))
axs[0].plot(df_stocks['AAPL'], label='AAPL')
axs[0].set_title('(a) AAPL', loc='left')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Stock price')
axs[1].plot(df_stocks['AMZN'], label='AMZN')
axs[1].set_title('(b) AMZN', loc='left')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Stock price')
plt.tight_layout()



B2_Ch3_7_B.py 
# calculate log returns
stock_return = []
for i in range(ticker_num):  
    return_tmp = np.log(df_stocks[[tickers[i]]]/df_stocks[[tickers[i]]].shift(1))[1:]  
    return_tmp = (return_tmp+1).cumprod()
    stock_return.append(return_tmp[[tickers[i]]])
    returns = pd.concat(stock_return,axis=1)
returns.head()

# calculate mu and sigma
mu = returns.mean()
sigma = returns.cov()

# cholesky decomp
R = np.linalg.cholesky(returns.corr())



B2_Ch3_7_C.py 
# parameters
T = 1
N = 252
Stock_0 = df_stocks.iloc[0]
dim = np.size(Stock_0)
t = np.linspace(0., T, int(N))
stockPrice = np.zeros([dim, int(N)])
stockPrice[:, 0] = Stock_0

# monte carlo simulations
MC_num = 100
mpl.style.use('ggplot')
fig, axs = plt.subplots(2, 1, figsize=(14,8))
for num in range(MC_num):
    for i in range(1, int(N)):    
        drift = (mu - 0.5 * np.diag(sigma)) * (t[i] - t[i-1])
        Z = np.random.normal(0., 1., dim)
        diffusion = np.matmul(Z, R) * (np.sqrt(t[i] - t[i-1]))
        stockPrice[:, i] = stockPrice[:, i-1]*np.exp(drift + diffusion)  
    axs[0].plot(t, stockPrice.T[:,0], label='AAPL')
    axs[0].set_title('(a) AAPL', loc='left')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Stock price')
    axs[1].plot(t, stockPrice.T[:,1], label='AMZN')
    axs[1].set_title('(b) AMZN', loc='left')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Stock price')
    plt.tight_layout()
    num+=1
