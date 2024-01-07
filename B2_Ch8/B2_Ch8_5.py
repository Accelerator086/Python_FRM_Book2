# B2_Ch8_5.py

###############
# Prepared by Ran An, Wei Lu, and Feng Zhang
# Editor-in-chief: Weisheng Jiang, and Sheng Tu
# Book 2  |  Financial Risk Management with Python
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2021
###############

B2_Ch8_5_A.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader
import seaborn as sns
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
df_stocks.columns = tickers



B2_Ch8_5_B.py
# cumulative returns
stock_return = []
for i in range(ticker_num):  
    return_tmp = np.log(df_stocks[[tickers[i]]]/df_stocks[[tickers[i]]].shift(1))[1:]  
    return_tmp = (return_tmp+1).cumprod()
    stock_return.append(return_tmp[[tickers[i]]])
    return_all = pd.concat(stock_return,axis=1)
return_all.head()



B2_Ch8_5_C.py
# plot cumulative returns of all stocks
plt.style.use('ggplot')
for i, col in enumerate(return_all.columns):
    return_all[col].plot()
plt.title('Cumulative returns')
plt.xlabel('Date')
plt.ylabel('Return')
plt.xticks(rotation=30)
plt.legend(return_all.columns)



B2_Ch8_5_D.py
# lastest return and price values
latest_return = return_all.iloc[-1,:]
latest_price = df_stocks.iloc[-1,:]
sigma = latest_return.std()

# weights for stocks in the portfolio
stock_weight = [0.2, 0.3, 0.1, 0.15, 0.25] 

# calculate expected return
expected_return = latest_return.dot(stock_weight)
print('The weighted expected portfolio return: %.2f' % expected_return)

# calculate weighted price
price = latest_price.dot(stock_weight)
print('The weighted price of the portfolio: %.0f' % price)



B2_Ch8_5_E.py
# monte carlo simulation
MC_num = 500
confidence_level = 0.95
time_step = 1440
for i in range(MC_num):  
  daily_returns = np.random.normal(expected_return/time_step, sigma/np.sqrt(time_step), time_step)
  plt.plot(daily_returns)
plt.axhline(np.percentile(daily_returns,(1.0-confidence_level)*100), color='r', linestyle='dashed')
plt.axhline(np.percentile(daily_returns,confidence_level*100), color='g', linestyle='dashed')
plt.axhline(np.mean(daily_returns), color='b', linestyle='solid')
plt.xlabel('Time')
plt.ylabel('Return')



B2_Ch8_5_F.py
# plot return distribution
sns.distplot(daily_returns, kde=True, color='lightblue')
plt.axvline(np.percentile(daily_returns,(1.0-confidence_level)*100), color='red', linestyle='dashed', linewidth=2)
plt.title("Return distribution")
plt.xlabel('Return')
plt.ylabel('Frequency')
plt.show()


B2_Ch8_5_G.py
initial_investment  = 1000000
VaR = initial_investment*np.percentile(daily_returns,(1.0-confidence_level)*100)
print('The value at risk is %.0f' % VaR)
