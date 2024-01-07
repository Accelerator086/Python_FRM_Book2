
###############
# Prepared by Ran An, Wei Lu, and Feng Zhang
# Editor-in-chief: Weisheng Jiang, and Sheng Tu
# Book 2  |  Financial Risk Management with Python
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2021
###############

from numpy import array, sqrt, dot, linspace, append, zeros_like, ones_like, size, identity
from pandas import read_excel, DataFrame, to_datetime
from qpsolvers import solve_qp
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
Weight_GMVP=solve_qp(
    CovarianceMatrix.to_numpy(),
    zeros_like(Singlename_Mean),
    -identity(size(Singlename_Mean)),
    zeros_like(Singlename_Mean),
    ones_like(Singlename_Mean),
    array([1.]),solver='daqp')

Port_Vol_GMVP = sqrt(dot(dot(Weight_GMVP,CovarianceMatrix.to_numpy()),Weight_GMVP))
Port_Return_GMVP = dot(Weight_GMVP,Singlename_Mean.to_numpy())

#%% bar chart GMVP weight
fig,ax=plt.subplots()

ax.barh(tickers,Weight_GMVP)
ax.set(xlabel='GMVP Weight Allocation',ylabel='Names')


#%% MVP portfolio, fixed return
Port_Return = 0.1
Weight_MVP=solve_qp(
    CovarianceMatrix.to_numpy(),
    zeros_like(Singlename_Mean), 
    -identity(size(Singlename_Mean)),
    zeros_like(Singlename_Mean),
    array([ones_like(Singlename_Mean),Singlename_Mean.to_numpy()]),
    array([1.,Port_Return]).reshape(2,),solver='daqp')


#%% Efficient Frontier

# =============================================================================
# Efficient Frontier
# =============================================================================
EF_vol = array([])
#Rp_range =  linspace(0.07,0.3, num=24)
Rp_range =  linspace(Port_Return_GMVP,max(Singlename_Mean), num=15)

for Rp in Rp_range:
    Weight_MVP=solve_qp(
        CovarianceMatrix.to_numpy(),
        zeros_like(Singlename_Mean), 
        -identity(size(Singlename_Mean)),
        zeros_like(Singlename_Mean),
        array([ones_like(Singlename_Mean),Singlename_Mean.to_numpy()]),
        array([1.,Rp]).reshape(2,),solver='daqp')
    Port_vol = sqrt(dot(dot(Weight_MVP,CovarianceMatrix.to_numpy()),Weight_MVP))
    EF_vol = append(EF_vol,array(Port_vol))

# =============================================================================
# In-efficient
# =============================================================================
InEF_vol = array([])
Rp_range_inEF =  linspace(min(Singlename_Mean),Port_Return_GMVP, num=8)

for Rp in Rp_range_inEF:
    Weight_MVP=solve_qp(
        CovarianceMatrix.to_numpy(),
        zeros_like(Singlename_Mean), 
        -identity(size(Singlename_Mean)),
        zeros_like(Singlename_Mean),
        array([ones_like(Singlename_Mean),Singlename_Mean.to_numpy()]),
        array([1.,Rp]).reshape(2,),solver='daqp')  
    Port_vol = sqrt(dot(dot(Weight_MVP,CovarianceMatrix.to_numpy()),Weight_MVP))
    InEF_vol = append(InEF_vol,array(Port_vol))
# =============================================================================
# Hyperbola curve
# =============================================================================
Hcurve_vol = array([])
Rp_range_Hcurve =  linspace(min(Singlename_Mean),max(Singlename_Mean), num=50)

for Rp in Rp_range_Hcurve:
    Weight_MVP=solve_qp(
        CovarianceMatrix.to_numpy(),
        zeros_like(Singlename_Mean), 
        -identity(size(Singlename_Mean)),
        zeros_like(Singlename_Mean),
        array([ones_like(Singlename_Mean),Singlename_Mean.to_numpy()]),
        array([1.,Rp]).reshape(2,),solver='daqp')  
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



