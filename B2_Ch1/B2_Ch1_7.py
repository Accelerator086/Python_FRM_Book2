# B2_Ch1_7.py

###############
# Prepared by Ran An, Wei Lu, and Feng Zhang
# Editor-in-chief: Weisheng Jiang, and Sheng Tu
# Book 2  |  Financial Risk Management with Python
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2021
###############

B2_Ch1_7_A.py
import numpy as np
import pandas_datareader
import matplotlib.pyplot as plt 
from arch import arch_model
import yfinance as yf
yf.pdr_override()

# sp500 price
sp500 = pandas_datareader.data.get_data_fred(['sp500'], start='12-28-2009', end='12-28-2020')

# daily log return
log_return_daily = np.log(sp500 / sp500.shift(1))
log_return_daily.dropna(inplace=True)

# ARCH(1) model
arch=arch_model(y=log_return_daily,mean='Constant',lags=0,vol='ARCH',p=1,o=0,q=0,dist='normal')
archmodel=arch.fit()
archmodel.summary()
archmodel.plot()

B2_Ch1_7_B.py
plt.figure(figsize=(12,8))
plt.plot(log_return_daily,label='Daily return')
plt.plot(archmodel.conditional_volatility, label='Conditional volatility')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Return/Volatility')

B2_Ch1_7_C.py
# GARCH(1,1) model
garch=arch_model(y=log_return_daily,mean='Constant',lags=0,vol='GARCH',p=1,o=0,q=1,dist='normal')
garchmodel=garch.fit()
garchmodel.summary()
garchmodel.plot()

B2_Ch1_7_D.py
plt.figure(figsize=(12,8))
plt.plot(log_return_daily,label='Daily return')
plt.plot(archmodel.conditional_volatility, label='Conditional volatility')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Return/Volatility')