# B2_Ch4_4.py

###############
# Prepared by Ran An, Wei Lu, and Feng Zhang
# Editor-in-chief: Weisheng Jiang, and Sheng Tu
# Book 2  |  Financial Risk Management with Python
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2021
###############

B2_Ch4_4_A.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error, r2_score


B2_Ch4_4_B.py
# read data
df = pd.read_csv(r'.\MultiLrRegrData.csv')
df.head()


B2_Ch4_4_C.py
# plot stock index price vs interest rate and unemployment rate
mpl.style.use('ggplot')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
ax1.scatter(df['InterestRate'], df['StockIndexPrice'], color='red')
ax1.set_title('(a) Stock index price VS interest rate', loc='left', fontsize=14)
ax1.set_xlabel('Interest rate', fontsize=14)
ax1.set_ylabel('Stock index price', fontsize=14)
ax1.set_yticks([700, 900, 1100, 1300, 1500])
ax1.grid(True)

ax2.scatter(df['UnemploymentRate'], df['StockIndexPrice'], color='green')
ax2.set_title('(b) Stock index price VS unemployment rate', loc='left', fontsize=14)
ax2.set_xlabel('Unemployment rate', fontsize=14)
ax2.set_ylabel('Stock index price', fontsize=14)
ax2.grid(True)


B2_Ch4_4_D.py
# implement linear regression model
x = df[['InterestRate','UnemploymentRate']] 
y = df['StockIndexPrice']
MultiLrModel = LinearRegression()
MultiLrModel.fit(x, y)

# plot multiple regression model
fig = plt.figure()
ax = plt.axes(projection='3d')
zdata = df['StockIndexPrice']
xdata = df['InterestRate']
ydata = df['UnemploymentRate']
ax.scatter(xdata, ydata, zdata, c=zdata)
x3d, y3d = np.meshgrid(xdata, ydata)
z3d_pred = MultiLrModel.intercept_+MultiLrModel.coef_[0]*x3d+MultiLrModel.coef_[1]*y3d
ax.plot_surface(x3d, y3d, z3d_pred, color = 'grey', rstride = 100, cstride = 100, alpha=0.3)
ax.set_title('Multiple Linear Regression', fontsize=14)
ax.set_xlabel('Interest rate')
ax.set_ylabel('Unemployment rate')
ax.set_zlabel('Stock index price')


B2_Ch4_4_E.py
zdata_pred = MultiLrModel.intercept_+MultiLrModel.coef_[0]*xdata+MultiLrModel.coef_[1]*ydata
rmse = (np.sqrt(mean_squared_error(zdata, zdata_pred)))
r2 = r2_score(zdata, zdata_pred)
print('RMSE of this polynomial regression model: %.2f' % rmse)
print('R square of this polynomial regression model: %.2f' % r2)
