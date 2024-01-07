
###############
# Prepared by Ran An, Wei Lu, and Feng Zhang
# Editor-in-chief: Weisheng Jiang, and Sheng Tu
# Book 2  |  Financial Risk Management with Python
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2021
###############

from numpy import sqrt, dot, zeros_like, ones_like
from pandas import read_excel, DataFrame, to_datetime
from scipy.optimize import minimize, LinearConstraint, Bounds
import matplotlib.pyplot as plt 

#%% Read data from excel
data = read_excel(r'.\Data_portfolio_1.xlsx')
data["Date"]=to_datetime(data["Date"])
data=data.set_index("Date")

#%% Return Vector, Volatility Vector, Variance-Covariance Matrix, Correlation Matrix
Singlename_Mean = DataFrame.mean(data)*12
Singlename_Vol = DataFrame.std(data)*sqrt(12)
CorrelationMatrix = DataFrame.corr(data)
CovarianceMatrix = DataFrame.cov(data)*12

#%% Scatter plot
tickers = Singlename_Mean.index.tolist()

fig,ax=plt.subplots()
ax.scatter(Singlename_Vol,Singlename_Mean,color="blue")

for x_pos, y_pos, label in zip(Singlename_Vol, Singlename_Mean, tickers):
    ax.annotate(label,             
                xy=(x_pos, y_pos), 
                xytext=(7, 0),     
                textcoords='offset points', 
                ha='left',         
                va='center')      

ax.set(xlabel='Portfolio Volatility',ylabel='Portfolio Return')


#%% define portfolio variance
w0= zeros_like(Singlename_Vol)
w0[1]=1

def MinVar(weight, *args):        
    CovMatrix = args
    
    obj = dot(dot(weight,CovMatrix),weight)
    return obj

#%% GMVP portfolio
linear_constraint = LinearConstraint(ones_like(Singlename_Vol),[1],[1])

res = minimize(MinVar, w0,
               args=(CovarianceMatrix.to_numpy()),
               method='trust-constr',
               constraints=[linear_constraint])

Weight_GMVP = res.x

Port_Vol_GMVP = sqrt(dot(dot(Weight_GMVP,CovarianceMatrix.to_numpy()),Weight_GMVP))
Port_Return_GMVP = dot(Weight_GMVP,Singlename_Mean.to_numpy())

#%% bar chart GMVP weight
fig,ax=plt.subplots()

ax.barh(tickers,Weight_GMVP)
ax.set(xlabel='GMVP Weight Allocation',ylabel='Names')


#%% GMVP portfolio w/o short
linear_constraint = LinearConstraint(ones_like(Singlename_Vol),[1],[1])
bounds = Bounds(zeros_like(Singlename_Vol), ones_like(Singlename_Vol))
res = minimize(MinVar, w0,
               args=(CovarianceMatrix.to_numpy()),
               method='trust-constr',
               bounds = bounds,
               constraints=[linear_constraint])

Weight_GMVP = res.x

Port_Vol_GMVP = sqrt(dot(dot(Weight_GMVP,CovarianceMatrix.to_numpy()),Weight_GMVP))
Port_Return_GMVP = dot(Weight_GMVP,Singlename_Mean.to_numpy())

#%% bar chart GMVP weight w/o short
fig,ax=plt.subplots()

ax.barh(tickers,Weight_GMVP)
ax.set(xlabel='GMVP Weight Allocation (w/o Short)',ylabel='Names')