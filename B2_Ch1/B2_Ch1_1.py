# B2_Ch1_1.py

###############
# Prepared by Ran An, Wei Lu, and Feng Zhang
# Editor-in-chief: Weisheng Jiang, and Sheng Tu
# Book 2  |  Financial Risk Management with Python
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2021
###############

B2_Ch1_1_A.py
import numpy as np
import pandas_datareader
# 由于Yahoo停止服务，采用yfinance包辅助：https://zhuanlan.zhihu.com/p/655866758 ， https://github.com/pydata/pandas-datareader/issues/952
import yfinance as yf
yf.pdr_override()

ticker = 'AMZN'
# startdate = datetime.datetime(2020,12,21);enddate = datetime.datetime(2020,12,28)
stock = pandas_datareader.data.get_data_yahoo(ticker, start='2020-12-21', end='2020-12-29')['Adj Close'] #经验证zhihu的yf.download效果等同，原因参见yf.pdr_override()解释
print(stock)

B2_Ch1_1_B.py
# via formula
returns_daily = (stock / stock.shift(1)) - 1
print(returns_daily)

B2_Ch1_1_C.py
# alternative via pct_change() function
returns_daily = stock.pct_change()
print(returns_daily)

B2_Ch1_1_D.py
# log return
log_return_daily = np.log(stock / stock.shift(1))
print(log_return_daily)