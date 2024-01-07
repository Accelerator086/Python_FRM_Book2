# B2_Ch4_7.py

###############
# Prepared by Ran An, Wei Lu, and Feng Zhang
# Editor-in-chief: Weisheng Jiang, and Sheng Tu
# Book 2  |  Financial Risk Management with Python
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2021
###############

B2_Ch4_7_A.py
# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib as mpl
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

B2_Ch4_6_B.py
data = pd.read_csv(r'.\PolyRegrData.csv')

# plot data
mpl.style.use('ggplot')
plt.figure(figsize=(14,8))
plt.scatter(data.iloc[:,0].values,data.iloc[:,1].values, c='#1f77b4')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Raw Data')


B2_Ch4_6_C.py
# preprocess input data
x = data.iloc[:,0].values.reshape(-1, 1)
y = data.iloc[:,1].values.reshape(-1, 1)
polynomial_features= PolynomialFeatures(degree=3)
x_poly = polynomial_features.fit_transform(x)
# create and then fit model
LRmodel = LinearRegression()
LRmodel.fit(x_poly,y)
print('intercept:', LRmodel.intercept_)
print('slope:', LRmodel.coef_)

# plot
plt.plot(x,y,'o',c='#1f77b4')
y_poly_pred = LRmodel.predict(x_poly)
plt.plot(x,y_poly_pred,'red')
plt.legend(['Raw Data',       
            'y=%5.2f+%5.2f*x+%5.2f*x²+%5.2f*x³' % (LRmodel.intercept_, LRmodel.coef_[0][1],LRmodel.coef_[0][2],LRmodel.coef_[0][3])    
            ], prop={'size': 8})
plt.title('Polynomial Regression Model')


B2_Ch4_6_D.py
# valuate model
rmse = np.sqrt(mean_squared_error(y,y_poly_pred))
r2 = r2_score(y,y_poly_pred)
print('RMSE of this polynomial regression model: %.2f' % rmse)
print('R square of this polynomial regression model: %.2f' % r2)
