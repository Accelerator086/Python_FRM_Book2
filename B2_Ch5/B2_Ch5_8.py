# B2_Ch5_8.py

###############
# Prepared by Ran An, Wei Lu, and Feng Zhang
# Editor-in-chief: Weisheng Jiang, and Sheng Tu
# Book 2  |  Financial Risk Management with Python
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2021
###############

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

def option_analytical(S0, vol, r, q, t, K, PutCall):
    d1 = (np.log(S0 / K) + (r - q + 0.5 * vol ** 2) * t) / (vol * np.sqrt(t))
    d2 = (np.log(S0 / K) + (r - q - 0.5 * vol ** 2) * t) / (vol * np.sqrt(t))
    price =  PutCall*S0 * np.exp(-q * t) * norm.cdf(PutCall*d1, 0.0, 1.0) - PutCall* K * np.exp(-r * t) * norm.cdf(PutCall*d2, 0.0, 1.0)  
        
    return price


def Binomialtree(n, S0, K, r, q, vol, t, PutCall, EuropeanAmerican,Tree):  
    deltaT = t/n 
    if Tree == 'CRR':
        u = np.exp(vol*np.sqrt(deltaT))
        d = 1./u
        p = (np.exp((r - q)*deltaT)-d) / (u-d) 
    elif Tree == 'JD':
        u = np.exp((r - q - vol**2*0.5)*deltaT + vol*np.sqrt(deltaT))
        d = np.exp((r - q - vol**2*0.5)*deltaT - vol*np.sqrt(deltaT))
        p = 0.5  
    elif Tree =='LR':
        def  h_function(z,n):
            h = 0.5+np.sign(z)*np.sqrt(0.25-0.25*np.exp(-((z/(n+1/3+0.1/(n+1)))**2)*(n+1/6)))
            return h
        
        if np.mod(n,2)>0:
            n_bar = n
        else:
            n_bar = n + 1
        
        d1 = (np.log(S0/K)+(r-q+vol**2/2)*t)/vol/np.sqrt(t);
        d2 = (np.log(S0/K)+(r-q-vol**2/2)*t)/vol/np.sqrt(t);
        pbar = h_function(d1,n_bar)
        p = h_function(d2,n_bar)
        u = np.exp((r-q)*deltaT)*pbar/p
        d = (np.exp((r-q)*deltaT)-p*u)/(1-p)    
            
    else:
        print("Tree type not supported")
    #Binomial price tree
    stockvalue = np.zeros((n+1,n+1))
    stockvalue[0,0] = S0
    for i in range(1,n+1):
        stockvalue[i,0] = stockvalue[i-1,0]*u
        for j in range(1,i+1):
            stockvalue[i,j] = stockvalue[i-1,j-1]*d

    #option value at final node   
    optionvalue = np.zeros((n+1,n+1))
    for j in range(n+1):
        if PutCall=="Call": # Call
            optionvalue[n,j] = max(0, stockvalue[n,j]-K)
        elif PutCall=="Put": #Put
            optionvalue[n,j] = max(0, K-stockvalue[n,j])
    if deltaT != 0: 
    #backward calculation for option price    
        for i in range(n-1,-1,-1):
            for j in range(i+1):
                if EuropeanAmerican=="American":
                    if PutCall=="Put":
                        optionvalue[i,j] = max(0, K-stockvalue[i,j], np.exp(-r*deltaT)*(p*optionvalue[i+1,j]+(1-p)*optionvalue[i+1,j+1]))
                    elif PutCall=="Call":
                        optionvalue[i,j] = max(0, stockvalue[i,j]-K, np.exp(-r*deltaT)*(p*optionvalue[i+1,j]+(1-p)*optionvalue[i+1,j+1]))
                    else:
                        print("PutCall type not supported")
                elif EuropeanAmerican=="European":    
                    if PutCall=="Put":
                        optionvalue[i,j] = max(0, np.exp(-r*deltaT)*(p*optionvalue[i+1,j]+(1-p)*optionvalue[i+1,j+1]))
                    elif PutCall=="Call":
                        optionvalue[i,j] = max(0, np.exp(-r*deltaT)*(p*optionvalue[i+1,j]+(1-p)*optionvalue[i+1,j+1]))
                    else:
                        print("PutCall type not supported")
                else:
                    print("Excercise type not supported")
    else:
        optionvalue[0,0] = optionvalue[n,j]     
                    
    return optionvalue[0,0]

    # Inputs
n = 20    #number of steps
S0 = 50  #initial underlying asset price
r = 0.01  #risk-free interest rate
q = 0.0 #dividend yield
K = 55   #strike price
vol = 0.3 #volatility
t = 1.0

bs_price = option_analytical(S0, vol, r, q, t, K, PutCall=1)
print('analytical Price: %.4f' % bs_price)

n= range(5, 300, 1)
prices_crr = np.array([Binomialtree(x, S0, K, r, q, vol, t, PutCall="Call", EuropeanAmerican="European",Tree = 'CRR') for x in n])
discrepancy_crr = (prices_crr/bs_price -1)/0.01 

plt.figure()
plt.plot(n, prices_crr,"b-",label='CRR',lw = 1)

plt.plot([5,300],[bs_price, bs_price], "r-", label='BSM',lw=1, alpha=0.6)
plt.xlabel("Number of steps")
plt.ylabel("Call option price, C (USD)")
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

n= range(5, 300, 2)
prices_lr = np.array([Binomialtree(x, S0, K, r, q, vol, t, PutCall="Call", EuropeanAmerican="European",Tree = 'LR') for x in n])
discrepancy_lr = (prices_lr/bs_price -1)/0.01 

plt.plot(n, prices_lr,"k-", label='LR',lw = 1)
plt.legend(loc='upper center')
