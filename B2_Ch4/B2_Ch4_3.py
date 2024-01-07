# B2_Ch4_3.py

###############
# Prepared by Ran An, Wei Lu, and Feng Zhang
# Editor-in-chief: Weisheng Jiang, and Sheng Tu
# Book 2  |  Financial Risk Management with Python
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2021
###############

import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import datetime as dt

# WTI price        
df_WTIPrice = pd.read_csv(r'.\WTI.csv', sep=',', usecols=['Date', 'Price'])
# hazard rate of an energy company
df_HazardRate = pd.read_csv(r'.\HazardRate.csv', sep=',', usecols=['Date', 'HazardRate'])
# merge harzard rate file and wti price file
df_dwr = df_HazardRate.merge(df_WTIPrice, left_on='Date', right_on='Date', how = 'inner')

df_dwr['Date'] = pd.to_datetime(df_dwr['Date'])
df_dwr = df_dwr[(df_dwr['Date']>=dt.datetime(2008, 8, 30))&(df_dwr['Date']<=dt.datetime(2008, 10, 30))]
xdata = df_dwr['Price']
ydata = df_dwr['HazardRate']
plt.plot(xdata, ydata, 'o', label='data')

# linear regression between hazard rate of an energy company and wti
slope, intercept, r_value, p_value, std_err = stats.linregress(xdata,ydata)
print('slope: %f, intercept: %f, r_value: %f, p_value: %f, std_err: %f' % (slope, intercept, r_value, p_value, std_err))

R_squared = r_value*r_value
print('R squared: %.2f' % R_squared)
rline = intercept + slope*xdata

plt.plot(xdata, rline,'r-')
plt.title('Simple Linear Regression')
plt.xlabel('WTI')
plt.ylabel('Hazard Rate')
plt.gca().set_yticks([0.005, 0.010, 0.015, 0.020])
plt.legend(['Observed Data', 'y=%5.4f+%5.5f*x, R²=%5.2f' % (intercept, slope, r_value**2)])

plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().yaxis.set_ticks_position('left')
plt.gca().xaxis.set_ticks_position('bottom')
