# B2_Ch1_10.py

###############
# Prepared by Ran An, Wei Lu, and Feng Zhang
# Editor-in-chief: Weisheng Jiang, and Sheng Tu
# Book 2  |  Financial Risk Management with Python
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2021
###############

B2_Ch1_10_A.py
from mibian import BS
import pandas as pd
import matplotlib.pyplot as plt

# convert data to dataframe and initialize "Implied volatility" column
# https://www.cboe.com/delayed_quotes/spx/quote_table
option_data = pd.read_csv(r".\SPX_Option.csv") # 这里原数据存在处理：call price为bid & ask的均值计算得到。
option_data['date'] = pd.to_datetime(option_data['date'])
option_data['Implied volatility'] = 0
option_data.head


B2_Ch1_10_B.py
# function to calculate implied volatility
def compute_implied_volatility(row):
    underlyingPrice = row['underlying value']
    strikePrice = row['strike']
    interestRate = 0.002
    daysToMaturity = row['days to maturity']
    optionPrice = row['call price']
    result = BS([underlyingPrice, strikePrice, interestRate, daysToMaturity], callPrice= optionPrice)
    return result.impliedVolatility

option_data['Implied volatility'] = option_data.apply(compute_implied_volatility, axis=1)


B2_Ch1_10_C.py
# plot volatility smile
option_data = option_data[option_data['date'] == pd.to_datetime('1/15/2021')]
plt.plot(option_data['strike'], option_data['Implied volatility'])
plt.title('Volatility smile')
plt.ylabel('Implied volatility')
plt.xlabel('Strike price')           

plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().yaxis.set_ticks_position('left')
plt.gca().xaxis.set_ticks_position('bottom')