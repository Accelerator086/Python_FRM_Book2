# B2_Ch1_8.py

###############
# Prepared by Ran An, Wei Lu, and Feng Zhang
# Editor-in-chief: Weisheng Jiang, and Sheng Tu
# Book 2  |  Financial Risk Management with Python
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2021
###############

import numpy as np
import pandas_datareader
import matplotlib.pyplot as plt 
import datetime
import matplotlib.dates as mdates
import yfinance as yf
yf.pdr_override()

# sp500 price
sp500 = pandas_datareader.data.get_data_fred(['sp500'], start='12-28-2009', end='12-28-2020')

# daily log return
log_return_daily = np.log(sp500 / sp500.shift(1))
log_return_daily.dropna(inplace=True)

n = 250
r = log_return_daily.iloc[-n:]

# volatility prediction by EWMA with λ=0.94
lmd = 0.94
vol_ewma = np.zeros(n)
vol_ewma[0] = log_return_daily[(-n+1):(-n+6)].std()
for i in range(n-1):
    vol_ewma[i+1] = np.sqrt(lmd*vol_ewma[i]**2 + (1-lmd)*r.iloc[i]**2)
    
# volatility prediction by ARCH(1)
omega_arch = 0.000068
alpha1 = 0.45 
vol_arch = np.zeros(n)
vol_arch[0] = np.sqrt(omega_arch + alpha1*log_return_daily.iloc[-n-1]**2)
for i in range(n-1):
    vol_arch[i+1] = np.sqrt(omega_arch + alpha1*r.iloc[i]**2)

#GARCH(1,1)
omega = 0.000002
alpha1 = 0.2
beta1 = 0.78

vol_garch = np.zeros(n)
vol_garch[0] = log_return_daily[-n+1:-n+6].std()
for i in range(n-1):
    vol_garch[i+1] = np.sqrt(omega + alpha1*r.iloc[i]**2 + beta1*vol_garch[i]**2)

# plot the curves
xdate=(r.index+datetime.timedelta(days=1))
plt.figure(figsize=(12,8))
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.title('Volatility comparison')
plt.plot(xdate,vol_arch, label='ARCH(1)')
plt.plot(xdate,vol_garch, label='GARCH(1,1)')
plt.plot(xdate,vol_ewma, label='EWMA') 

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(15))
plt.xticks(rotation=30)
plt.legend()
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().yaxis.set_ticks_position('left')
plt.gca().xaxis.set_ticks_position('bottom')
