
###############
# Prepared by Ran An, Wei Lu, and Feng Zhang
# Editor-in-chief: Weisheng Jiang, and Sheng Tu
# Book 2  |  Financial Risk Management with Python
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2021
###############

from numpy import array, sqrt, dot, linspace, ones, zeros, size, append
from pandas import read_excel, DataFrame, to_datetime
from numpy.linalg import inv

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


#%% GMVP portfolio
CalMat = ones((size(Singlename_Mean)+1,size(Singlename_Mean)+1))
CalMat[0:-1,0:-1] = 2*CovarianceMatrix.to_numpy()
CalMat[0:-1,-1] = - CalMat[0:-1,-1]
CalMat[-1,-1] = 0.0

Vec1 = zeros((size(Singlename_Mean)+1))
Vec1[-1] = 1

SolutionVec1 = dot(inv(CalMat),Vec1)

Weight_GMVP = SolutionVec1[0:-1]

Port_Vol_GMVP = sqrt(dot(dot(Weight_GMVP,CovarianceMatrix.to_numpy()),Weight_GMVP))
Port_Return_GMVP = dot(Weight_GMVP,Singlename_Mean.to_numpy())

#%% bar chart GMVP weight
fig,ax=plt.subplots()

ax.barh(tickers,Weight_GMVP)
ax.set(xlabel='GMVP Weight Allocation',ylabel='Names')

#%% MVP portfolio, fixed return
Port_Return = 0.30
CalMat = ones((size(Singlename_Mean)+2,size(Singlename_Mean)+2))
CalMat[0:-2,0:-2] = 2*CovarianceMatrix.to_numpy()
CalMat[0:-2,-2] = - CalMat[0:-2,-2]
CalMat[0:-2,-1] = - Singlename_Mean.to_numpy()
CalMat[-1,0:-2] = Singlename_Mean.to_numpy()
CalMat[-2:,-2:] = zeros((2,2))

Vec2 = zeros((size(Singlename_Mean)+2))
Vec2[-2] = 1
Vec2[-1] = Port_Return

SolutionVec2 = dot(inv(CalMat),Vec2)

Weight_MVP = SolutionVec2[0:-2]

#%% Efficient Frontier

CalMat = ones((size(Singlename_Mean)+2,size(Singlename_Mean)+2))
CalMat[0:-2,0:-2] = 2*CovarianceMatrix.to_numpy()
CalMat[0:-2,-2] = - CalMat[0:-2,-2]
CalMat[0:-2,-1] = - Singlename_Mean.to_numpy()
CalMat[-1,0:-2] = Singlename_Mean.to_numpy()
CalMat[-2:,-2:] = zeros((2,2))
Vec2 = zeros((size(Singlename_Mean)+2))
Vec2[-2] = 1

# =============================================================================
# Efficient Frontier
# =============================================================================
EF_vol = array([])
#Rp_range =  linspace(0.07,0.3, num=24)
Rp_range =  linspace(Port_Return_GMVP,0.3, num=25)

for Rp in Rp_range:    
    Vec2[-1] = Rp    
    SolutionVec2 = dot(inv(CalMat),Vec2)
    
    Weight_MVP = SolutionVec2[0:-2]
    
    Port_vol = sqrt(dot(dot(Weight_MVP,CovarianceMatrix.to_numpy()),Weight_MVP))
    EF_vol = append(EF_vol,array(Port_vol))

# =============================================================================
# In-efficient
# =============================================================================
InEF_vol = array([])
Rp_range_inEF =  linspace(0.0,Port_Return_GMVP, num=10)

for Rp in Rp_range_inEF:    
    Vec2[-1] = Rp    
    SolutionVec2 = dot(inv(CalMat),Vec2)
    
    Weight_MVP = SolutionVec2[0:-2]
    
    Port_vol = sqrt(dot(dot(Weight_MVP,CovarianceMatrix.to_numpy()),Weight_MVP))
    InEF_vol = append(InEF_vol,array(Port_vol))
# =============================================================================
# Hyperbola curve
# =============================================================================
Hcurve_vol = array([])
Rp_range_Hcurve =  linspace(0.0,0.3, num=100)

for Rp in Rp_range_Hcurve:    
    Vec2[-1] = Rp    
    SolutionVec2 = dot(inv(CalMat),Vec2)
    
    Weight_MVP = SolutionVec2[0:-2]
    
    Port_vol = sqrt(dot(dot(Weight_MVP,CovarianceMatrix.to_numpy()),Weight_MVP))
    Hcurve_vol = append(Hcurve_vol,array(Port_vol))

#%% plot Efficient Frontier portfolios
fig,ax=plt.subplots()
ax.plot(Hcurve_vol,Rp_range_Hcurve)
ax.scatter(Port_Vol_GMVP,Port_Return_GMVP, marker='^')
ax.scatter(InEF_vol,Rp_range_inEF)
ax.scatter(EF_vol,Rp_range)
ax.scatter(Singlename_Vol,Singlename_Mean,color="blue")

for x_pos, y_pos, label in zip(Singlename_Vol, Singlename_Mean, tickers):
    ax.annotate(label,             
                xy=(x_pos, y_pos), 
                xytext=(7, 0),     
                textcoords='offset points', 
                ha='left',         
                va='center')  

ax.set(xlabel='Portfolio Volatility',ylabel='Portfolio Return')







