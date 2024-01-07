# B2_Ch8_4.py

###############
# Prepared by Ran An, Wei Lu, and Feng Zhang
# Editor-in-chief: Weisheng Jiang, and Sheng Tu
# Book 2  |  Financial Risk Management with Python
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2021
###############

B2_Ch8_4_A.py
import matplotlib.pyplot as plt
import numpy as np
import pandas_datareader
import scipy.stats as stats
import tabulate   
import yfinance as yf
yf.pdr_override() 
 
# prices = pandas_datareader.DataReader('AAPL', start='2015-11-30', end = '2020-11-30', data_source='yahoo')  
prices = pandas_datareader.data.get_data_yahoo('AAPL', start='2015-11-30', end = '2020-11-30')    
df_stocks = prices[['Adj Close']]
    
# stock returns
returns = np.log(df_stocks/df_stocks.shift(1))
returns = returns.dropna()



B2_Ch8_4_B.py
# historical VaR
returns.sort_values('Adj Close', ascending=True, inplace=True)
HistVaR_90 = returns.quantile(0.1, interpolation='lower')[0]
HistVaR_95 = returns.quantile(0.05, interpolation='lower')[0]
HistVaR_99 = returns.quantile(0.01, interpolation='lower')[0]
print(tabulate.tabulate([['90%', HistVaR_90], ['95%', HistVaR_95], ['99%', HistVaR_99]], headers=['Confidence level', 'Value at Risk']))


B2_Ch8_4_C.py
# parameteric VaR
mu = np.mean(returns['Adj Close'])
std = np.std(returns['Adj Close'])
ParaVaR_90 = stats.norm.ppf(0.1, mu, std)
ParaVaR_95 = stats.norm.ppf(0.05, mu, std)
ParaVaR_99 = stats.norm.ppf(0.01, mu, std)
print(tabulate.tabulate([['90%', ParaVaR_90], ['95%', ParaVaR_95], ['99%', ParaVaR_99]], headers=['Confidence level', 'Value at Risk']))



B2_Ch8_4_D.py
# plot distribution 
plt.style.use('ggplot')
fig, ax = plt.subplots(1,1, figsize=(12,6))
x = np.linspace(mu-5*std, mu+5*std, 500)
ax.hist(returns['Adj Close'], bins=100, density=True, histtype="stepfilled", alpha=0.5)
ax.axvline(HistVaR_95, ymin=0, ymax=0.2, color='g', ls=':', alpha=0.7, label='95% historical VaR')
ax.axvline(ParaVaR_95, ymin=0, ymax=0.2, color='b', ls=':', alpha=0.7, label='95% parametric VaR')
ax.plot(x, stats.norm.pdf(x, mu, std))
ax.legend()
ax.set_title("Return distribution")
ax.set_xlabel("Return")
ax.set_ylabel("Frequency")
