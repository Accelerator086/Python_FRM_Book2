
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

#%% GMVP portfolio w short
Weight_GMVP_wShort=solve_qp(
    CovarianceMatrix.to_numpy(),
    zeros_like(Singlename_Mean),
    None,None,
    ones_like(Singlename_Mean),
    array([1.]),solver='daqp')

Port_Vol_GMVP_wShort = sqrt(dot(dot(Weight_GMVP_wShort,CovarianceMatrix.to_numpy()),Weight_GMVP_wShort))
Port_Return_GMVP_wShort = dot(Weight_GMVP_wShort,Singlename_Mean.to_numpy())

#%% GMVP portfolio wo short
Weight_GMVP_woShort=solve_qp(
    CovarianceMatrix.to_numpy(),
    zeros_like(Singlename_Mean),
    -identity(size(Singlename_Mean)),
    zeros_like(Singlename_Mean),
    ones_like(Singlename_Mean),
    array([1.]),solver='daqp')

Port_Vol_GMVP_woShort = sqrt(dot(dot(Weight_GMVP_woShort,CovarianceMatrix.to_numpy()),Weight_GMVP_woShort))
Port_Return_GMVP_woShort = dot(Weight_GMVP_woShort,Singlename_Mean.to_numpy())

#%% Efficient Frontier
# =============================================================================
# Hyperbola curve 
# =============================================================================
Hcurve_vol = array([])
Rp_range_Hcurve =  linspace(0.00,0.2, num=200)

for Rp in Rp_range_Hcurve:
    Weight_MVP=solve_qp(
        CovarianceMatrix.to_numpy(),
        zeros_like(Singlename_Mean), 
        None,None,
        array([ones_like(Singlename_Mean),Singlename_Mean.to_numpy()]),
        array([1.,Rp]).reshape(2,),solver='daqp')      
    Port_vol = sqrt(dot(dot(Weight_MVP,CovarianceMatrix.to_numpy()),Weight_MVP))
    Hcurve_vol = append(Hcurve_vol,array(Port_vol))
# =============================================================================
# Hyperbola curve - wo short
# =============================================================================
Hcurve_vol_woshort = array([])
Rp_range_Hcurve_woshort =  linspace(min(Singlename_Mean),max(Singlename_Mean), num=100)

for Rp in Rp_range_Hcurve_woshort:
    Weight_MVP=solve_qp(
        CovarianceMatrix.to_numpy(),
        zeros_like(Singlename_Mean), 
        -identity(size(Singlename_Mean)),
        zeros_like(Singlename_Mean),
        array([ones_like(Singlename_Mean),Singlename_Mean.to_numpy()]),
        array([1.,Rp]).reshape(2,),solver='daqp')  
    Port_vol = sqrt(dot(dot(Weight_MVP,CovarianceMatrix.to_numpy()),Weight_MVP))
    Hcurve_vol_woshort = append(Hcurve_vol_woshort,array(Port_vol))

#%% plot Efficient Frontier portfolios
tickers = Singlename_Mean.index.tolist()
fig,ax=plt.subplots()
ax.plot(Hcurve_vol,Rp_range_Hcurve)
ax.plot(Hcurve_vol_woshort,Rp_range_Hcurve_woshort)
#ax.scatter(Port_Vol_GMVP,Port_Return_GMVP, marker='^')
#ax.scatter(InEF_vol,Rp_range_inEF)
#ax.scatter(EF_vol,Rp_range)
ax.plot(Port_Vol_GMVP_wShort,Port_Return_GMVP_wShort,'^',label='GMVP w/ Short')
ax.plot(Port_Vol_GMVP_woShort,Port_Return_GMVP_woShort,'^',label='GMVP w/o Short')

ax.scatter(Singlename_Vol,Singlename_Mean,color="blue")

for x_pos, y_pos, label in zip(Singlename_Vol, Singlename_Mean, tickers):
    ax.annotate(label,             
                xy=(x_pos, y_pos), 
                xytext=(7, 0),     
                textcoords='offset points', 
                ha='left',         
                va='center')  
ax.set(xlabel='Portfolio Volatility',ylabel='Portfolio Return')
plt.legend()


#%% MVP portfolio, fixed return, w/o short
Port_Return = 0.30
Weight_MVP=solve_qp(
    CovarianceMatrix.to_numpy(),
    zeros_like(Singlename_Mean), 
    None,None,
    array([ones_like(Singlename_Mean),Singlename_Mean.to_numpy()]),
    array([1.,Port_Return]).reshape(2,),solver='daqp')

fig,ax=plt.subplots()
ax.barh(tickers,Weight_MVP)
ax.set(xlabel='Weight',ylabel='Names')

