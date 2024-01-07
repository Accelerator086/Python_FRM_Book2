
###############
# Prepared by Ran An, Wei Lu, and Feng Zhang
# Editor-in-chief: Weisheng Jiang, and Sheng Tu
# Book 2  |  Financial Risk Management with Python
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2021
###############

from numpy import sqrt, linspace, corrcoef, zeros
from pandas import read_excel, DataFrame, to_datetime

#%% Read data from excel
data = read_excel(r'.\Data_portfolio_2.xlsx')
data["Date"]=to_datetime(data["Date"])
data=data.set_index("Date")

#%% CAPM beta
Mean = DataFrame.mean(data)*12
Vol = DataFrame.std(data)*sqrt(12)

Singlename_Return = data.iloc[:,0:-2]

MktExcess = data.iloc[:,-2]
RF = data.iloc[:,-1]

Singlename_ExcessReturn = Singlename_Return

n= len(Singlename_Return.columns)

Correlation_v_Mkt = zeros(n)

for k in linspace(0,n-1,n):
    k=int(k)
    Singlename_ExcessReturn.iloc[:,k] = Singlename_Return.iloc[:,k] - RF
    Correlation_v_Mkt[k] = corrcoef(Singlename_ExcessReturn.iloc[:,k].to_numpy(),MktExcess.to_numpy())[1,0]
    

Vol_Excess = DataFrame.std(Singlename_ExcessReturn)*sqrt(12) 
Vol_Mkt = DataFrame.std(DataFrame(MktExcess))*sqrt(12) 


Beta = zeros(n)
Sys_exp_Vol = zeros(n)
Sys_exp_prct = zeros(n)

for k in linspace(0,n-1,n):
    k=int(k)
    Beta[k] = Correlation_v_Mkt[k]*Vol_Excess[k]/Vol_Mkt
    Sys_exp_Vol[k] = Beta[k]*Vol_Mkt
    Sys_exp_prct[k] = Sys_exp_Vol[k]**2/Vol_Excess[k]**2


