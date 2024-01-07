
###############
# Prepared by Ran An, Wei Lu, and Feng Zhang
# Editor-in-chief: Weisheng Jiang, and Sheng Tu
# Book 2  |  Financial Risk Management with Python
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2021
###############

from numpy import sqrt, linspace
import matplotlib.pyplot as plt

#%%  Specify Individual Assets' Returns, Volatilities, and Correlation

r1 = 0.11
r2 = 0.16
vol1 = 0.25
vol2 = 0.38

rho_range = [-1., -0.5, 0., 0.5, 1.]

#%%  Two Assets Mean-Variance Framework
def TwoAssetPort(w1,w2,r1,r2,sigma1,sigma2,rho):
    PortReturn = w1*r1 + w2*r2
    PortVol = sqrt((w1*sigma1)**2+(w2*sigma2)**2+2*w1*w2*sigma1*sigma2*rho)
    return PortReturn,PortVol

#%% Plot Return-Volatility
w1 = linspace(-0.3,1.5,190)
w2 = 1- w1

fig,ax=plt.subplots()

for rho in linspace(-1,1,17):
    TwoAssetPort_Return,TwoAssetPort_Vol = TwoAssetPort(w1,w2,r1,r2,vol1,vol2,rho)
    #ax.plot(TwoAssetPort_Vol,TwoAssetPort_Return,label='rho = '+str(int(rho*100))+'%')
    ax.plot(TwoAssetPort_Vol,TwoAssetPort_Return)

ax.set(xlabel='Portfolio Volatility',ylabel='Portfolio Return')


#%% find GMVP of a Two-asset portfolio
def GMVP_TwoAssetPort(r1,r2,sigma1,sigma2,rho):
    w1_star = (sigma2**2-rho*sigma1*sigma2)/(sigma1**2-2*rho*sigma1*sigma2+sigma2**2)
    w2_star = 1-w1_star
    PortReturn = w1_star*r1 + w2_star*r2
    PortVol = sqrt((w1_star*sigma1)**2+(w2_star*sigma2)**2+2*w1_star*w2_star*sigma1*sigma2*rho)
    return PortReturn,PortVol,w1_star,w2_star

for rho in rho_range:
    GMVP_return,GMVP_Vol,w1_star,w2_star = GMVP_TwoAssetPort(r1,r2,vol1,vol2,rho)
    print(rho,GMVP_return,GMVP_Vol,w1_star,w2_star)

#%% 
fig,ax=plt.subplots()

for rho in rho_range[0:4]:

    GMVP_return,GMVP_Vol,w1_star,w2_star = GMVP_TwoAssetPort(r1,r2,vol1,vol2,rho)
    
    if r1 < r2:
        w1_under = linspace(w1_star,1.5,100)
        w2_under = 1 - w1_under
        w1 = linspace(-0.3,w1_star,100)
        w2 = 1 - w1
    else:
        w1 = linspace(w1_star,1.5,100)
        w2 = 1 - w1
        w1_under = linspace(-0.3,w1_star,100)
        w2_under = 1 - w1_under    
    
    TwoAssetPort_Return_under,TwoAssetPort_Vol_under = TwoAssetPort(w1_under,w2_under,r1,r2,vol1,vol2,rho)
    TwoAssetPort_Return,TwoAssetPort_Vol = TwoAssetPort(w1,w2,r1,r2,vol1,vol2,rho)
    
    ax.plot(TwoAssetPort_Vol,TwoAssetPort_Return,'-', 
            TwoAssetPort_Vol_under,TwoAssetPort_Return_under,'--',
            #label='rho = '+str(int(rho*100))+'%')
            )
    ax.plot(GMVP_Vol,GMVP_return,'o')

ax.set(xlabel='Portfolio Volatility',ylabel='Portfolio Return')


#%% Plot Weight 1 vs Return & Volatility
w1 = linspace(-1.0,2.8,190)
w2 = 1- w1

fig,ax=plt.subplots()

for rho in rho_range:
    TwoAssetPort_Return,TwoAssetPort_Vol = TwoAssetPort(w1,w2,r1,r2,vol1,vol2,rho)
    ax.plot(w1,TwoAssetPort_Vol,label='rho = '+str(int(rho*100))+'%')

ax.set(xlabel='Weight 1',ylabel='Portfolio Volatility')
plt.legend()


fig,ax=plt.subplots()
ax.plot(w1,TwoAssetPort_Return)
ax.set(xlabel='Weight 1',ylabel='Portfolio Return')
plt.legend()


#%% Plot Two-Asset GMVP
from numpy import matrix, dot, ones, array, linspace, append, sqrt
from numpy.linalg import inv
import matplotlib.pyplot as plt

r1 = 0.11
r2 = 0.16
vol1 = 0.25
vol2 = 0.38

rho_range = linspace(-1,1,500)
Vol_GMVP_range = array([])
R_GMVP_range = array([])

for rho in rho_range:    
    CovM = matrix([[vol1**2, rho*vol1*vol2],[rho*vol1*vol2, vol2**2]])
    Var_GMVP = 1/dot(dot(ones(2),inv(CovM)),ones((2,1)))
    
    Vol_GMVP = sqrt(Var_GMVP)
    R_GMVP = Var_GMVP*dot(dot(ones(2),inv(CovM)),array([[r1],[r2]]))
    
    Vol_GMVP_range = append(Vol_GMVP_range,Vol_GMVP)
    R_GMVP_range = append(R_GMVP_range,R_GMVP)
    
fig,ax=plt.subplots()
ax.plot(rho_range,R_GMVP_range)
ax.set(xlabel='Correlation',ylabel='Portfolio Return')

fig,ax=plt.subplots()
ax.plot(rho_range,Vol_GMVP_range)
ax.set(xlabel='Correlation',ylabel='Portfolio Vol')

fig,ax=plt.subplots()
ax.plot(Vol_GMVP_range,R_GMVP_range,'o')
ax.set(xlabel='Portfolio Vol',ylabel='Portfolio Return')
    