
###############
# Prepared by Ran An, Wei Lu, and Feng Zhang
# Editor-in-chief: Weisheng Jiang, and Sheng Tu
# Book 2  |  Financial Risk Management with Python
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2021
###############

from numpy import array, sqrt, dot, linspace, append, zeros_like, ones_like
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

#%% Define Risk Free asset
RF = 0.02

#%% Scatter plot
tickers = Singlename_Mean.index.tolist()

fig,ax=plt.subplots()
ax.scatter(Singlename_Vol,Singlename_Mean,color="blue")
ax.scatter(0,RF,color="red")

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
    None,None,
    ones_like(Singlename_Mean),
    array([1.]),solver='daqp')

Port_Vol_GMVP = sqrt(dot(dot(Weight_GMVP,CovarianceMatrix.to_numpy()),Weight_GMVP))
Port_Return_GMVP = dot(Weight_GMVP,Singlename_Mean.to_numpy())

#%% bar chart GMVP weight
fig,ax=plt.subplots()

ax.barh(tickers,Weight_GMVP)
ax.set(xlabel='GMVP Weight Allocation',ylabel='Names')

#%% MVP portfolio, fixed return
Port_Return = 0.30
Weight_MVP=solve_qp(
    CovarianceMatrix.to_numpy(),
    zeros_like(Singlename_Mean), 
    None,None,
    array([ones_like(Singlename_Mean),Singlename_Mean.to_numpy()]),
    array([1.,Port_Return]).reshape(2,),solver='daqp')

fig,ax=plt.subplots()
tickers = Singlename_Mean.index.tolist()
ax.barh(tickers,Weight_MVP)
ax.set(xlabel='Weight',ylabel='Names')


#%% Efficient Frontier

# =============================================================================
# Efficient Frontier
# =============================================================================
EF_vol = array([])
#Rp_range =  linspace(0.07,0.3, num=24)
Rp_range =  linspace(Port_Return_GMVP,0.3, num=25)

for Rp in Rp_range:
    Weight_MVP=solve_qp(
        CovarianceMatrix.to_numpy(),
        zeros_like(Singlename_Mean), 
        None,None,
        array([ones_like(Singlename_Mean),Singlename_Mean.to_numpy()]),
        array([1.,Rp]).reshape(2,),solver='daqp')
    Port_vol = sqrt(dot(dot(Weight_MVP,CovarianceMatrix.to_numpy()),Weight_MVP))
    EF_vol = append(EF_vol,array(Port_vol))

# =============================================================================
# In-efficient
# =============================================================================
InEF_vol = array([])
Rp_range_inEF =  linspace(0.0,Port_Return_GMVP, num=10)

for Rp in Rp_range_inEF:
    Weight_MVP=solve_qp(
        CovarianceMatrix.to_numpy(),
        zeros_like(Singlename_Mean), 
        None,None,
        array([ones_like(Singlename_Mean),Singlename_Mean.to_numpy()]),
        array([1.,Rp]).reshape(2,),solver='daqp')    
    Port_vol = sqrt(dot(dot(Weight_MVP,CovarianceMatrix.to_numpy()),Weight_MVP))
    InEF_vol = append(InEF_vol,array(Port_vol))
# =============================================================================
# Hyperbola curve
# =============================================================================
Hcurve_vol = array([])
Rp_range_Hcurve =  linspace(0.001,0.3, num=100)

for Rp in Rp_range_Hcurve:
    Weight_MVP=solve_qp(
        CovarianceMatrix.to_numpy(),
        zeros_like(Singlename_Mean), 
        None,None,
        array([ones_like(Singlename_Mean),Singlename_Mean.to_numpy()]),
        array([1.,Rp]).reshape(2,),solver='daqp')      
    Port_vol = sqrt(dot(dot(Weight_MVP,CovarianceMatrix.to_numpy()),Weight_MVP))
    Hcurve_vol = append(Hcurve_vol,array(Port_vol))

#%% Optimal Risky portfolio
Er_initial = 0.15
Solution=solve_qp(
    CovarianceMatrix.to_numpy(),
    zeros_like(Singlename_Mean),
    None,None,
    array([Singlename_Mean.to_numpy()-RF]),
    array([Er_initial-RF]),solver='daqp')

Weight_ORP = Solution/sum(Solution)

Port_Vol_ORP = sqrt(dot(dot(Weight_ORP,CovarianceMatrix.to_numpy()),Weight_ORP))
Port_Return_ORP = dot(Weight_ORP,Singlename_Mean.to_numpy())

SR = (Port_Return_ORP-RF)/Port_Vol_ORP

#%% bar chart ORP weight
fig,ax=plt.subplots()

ax.barh(tickers,Weight_ORP)
ax.set(xlabel='Optimal Risk Portfolio Weight Allocation',ylabel='Names')


#%% Capital Market Line
vol_range = linspace(0,0.35,100)
CML = RF + SR*vol_range

#%% plot Efficient Frontier portfolios
fig,ax=plt.subplots()
ax.plot(Hcurve_vol,Rp_range_Hcurve)
ax.scatter(Port_Vol_GMVP,Port_Return_GMVP, marker='x')
#ax.scatter(InEF_vol,Rp_range_inEF)
#ax.scatter(EF_vol,Rp_range)
ax.scatter(0,RF,color="red")
ax.scatter(Port_Vol_ORP,Port_Return_ORP, marker='D',color="red")
ax.plot(vol_range,CML)
ax.scatter(Singlename_Vol,Singlename_Mean,color="blue")

for x_pos, y_pos, label in zip(Singlename_Vol, Singlename_Mean, tickers):
    ax.annotate(label,             
                xy=(x_pos, y_pos), 
                xytext=(7, 0),     
                textcoords='offset points', 
                ha='left',         
                va='center')  

ax.set(xlabel='Portfolio Volatility',ylabel='Portfolio Return')


#%% Optimal Indifference Utility Curve 1
A1 = 3
U_max_1 = RF + SR**2/(2*A1) 
Weight_P_1 = SR/(A1*Port_Vol_ORP) 

R1 = 1/2*A1*(vol_range**2) + U_max_1

E_c1 = RF+SR**2/A1
Vol_c1 = Weight_P_1*Port_Vol_ORP

#%% bar chart OCP1 weight
fig,ax=plt.subplots()

ax.barh(tickers,Weight_ORP*Weight_P_1)
ax.set(xlabel='Optimal Risk Portfolio Weight Allocation with A = '+ str(A1),ylabel='Names')

#%% Optimal Indifference Utility Curve 2
A2 = 5
U_max_2 = RF + SR**2/(2*A2)
Weight_P_2 = SR/(A2*Port_Vol_ORP) 

R2 = 1/2*A2*(vol_range**2) + U_max_2

E_c2 = RF+SR**2/A2
Vol_c2 = Weight_P_2*Port_Vol_ORP

#%% bar chart OCP2 weight
fig,ax=plt.subplots()

ax.barh(tickers,Weight_ORP*Weight_P_2)
ax.set(xlabel='Optimal Risk Portfolio Weight Allocation with A = '+ str(A2),ylabel='Names')

#%% plot Capital Market Line and Indifference Utility Curves
fig,ax=plt.subplots()
ax.plot(vol_range,CML)
ax.plot(vol_range,R1,color="green")
ax.plot(vol_range,R2,color="green")
ax.scatter(Port_Vol_ORP,Port_Return_ORP, marker='D')

ax.scatter(Vol_c1,E_c1, marker='*', color="purple")
ax.scatter(Vol_c2,E_c2, marker='*', color="purple")

ax.set(xlabel='Portfolio Volatility',ylabel='Portfolio Return')

#%% plot everything
fig,ax=plt.subplots()
ax.plot(Hcurve_vol,Rp_range_Hcurve)
ax.scatter(Port_Vol_GMVP,Port_Return_GMVP, marker='x')
#ax.scatter(InEF_vol,Rp_range_inEF)
#ax.scatter(EF_vol,Rp_range)
ax.scatter(0,RF,color="red")
ax.scatter(Port_Vol_ORP,Port_Return_ORP, marker='D')
ax.plot(vol_range,CML)

ax.plot(vol_range,R1,color="green")
ax.plot(vol_range,R2,color="green")

ax.scatter(Vol_c1,E_c1, marker='*', color="purple")
ax.scatter(Vol_c2,E_c2, marker='*', color="purple")

ax.scatter(Singlename_Vol,Singlename_Mean,color="blue")
for x_pos, y_pos, label in zip(Singlename_Vol, Singlename_Mean, tickers):
    ax.annotate(label,             
                xy=(x_pos, y_pos), 
                xytext=(7, 0),     
                textcoords='offset points', 
                ha='left',         
                va='center')  

ax.set(xlabel='Portfolio Volatility',ylabel='Portfolio Return')
