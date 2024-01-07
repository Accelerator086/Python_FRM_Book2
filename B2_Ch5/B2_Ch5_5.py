# B2_Ch5_5.py

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

def Binomialtree(n, S0, K, r, q, vol, t, PutCall, EuropeanAmerican):  
    deltaT = t/n 
    u = np.exp(vol*np.sqrt(deltaT))
    d = 1./u
    p = (np.exp((r-q)*deltaT)-d) / (u-d) 
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

def option_analytical(S0, vol, r, q, t, K, PutCall):
    d1 = (np.log(S0 / K) + (r - q + 0.5 * vol ** 2) * t) / (vol * np.sqrt(t))
    d2 = (np.log(S0 / K) + (r - q - 0.5 * vol ** 2) * t) / (vol * np.sqrt(t))
    
    price =  PutCall*S0 * np.exp(-q * t) * norm.cdf(PutCall*d1, 0.0, 1.0) - PutCall* K * np.exp(-r * t) * norm.cdf(PutCall*d2, 0.0, 1.0)  
        
    return price

    # Inputs
n= 2   #number of steps
S0 = 50  #initial underlying asset price
r = 0.03  #risk-free interest rate
q = 0.0 # dividend yield
K = 55   #strike price
vol = 0.3 #volatility
t = 2.0
PutCall = 1 # 1 for call;-1 for put

bs_price = option_analytical(S0, vol, r, q, t, K, PutCall=1)
print('analytical Price: %.4f' % bs_price)

n= range(2, 1012, 10)
prices = np.array([Binomialtree(x, S0, K, r, q, vol, t, PutCall="Call", EuropeanAmerican="European") for x in n])
discrepancy = (prices/bs_price -1)/0.01 

plt.figure()
plt.plot(n, prices,"-o",markersize = 2)
plt.plot([0,1012],[bs_price, bs_price], "r-", lw=2, alpha=0.6)
plt.xlabel("Number of steps")
plt.ylabel("Call option price, C (USD)")
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

plt.figure()
plt.plot(n, discrepancy,"-o",markersize = 2)
plt.plot([0,1012],[0, 0], "g-", lw=2, alpha=0.6)
plt.xlabel("Number of steps")
plt.ylabel("Discrepancy (%)")
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

bs_price = option_analytical(S0, vol, r, q, t, K, PutCall=-1)
print('analytical Price: %.4f' % bs_price)

n= range(2, 1012, 10)
prices = np.array([Binomialtree(x, S0, K, r, q, vol, t, PutCall="Put", EuropeanAmerican="European") for x in n])
discrepancy = (prices/bs_price -1)/0.01 

plt.figure()
plt.plot(n, prices,"-o",markersize = 2)
plt.plot([0,1012],[bs_price, bs_price], "r-", lw=2, alpha=0.6)
plt.xlabel("Number of steps")
plt.ylabel("Put option price, C (USD)")
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

plt.figure()
plt.plot(n, discrepancy,"-o",markersize = 2)
plt.plot([0,1012],[0, 0], "g-", lw=2, alpha=0.6)
plt.xlabel("Number of steps")
plt.ylabel("Discrepancy (%)")
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)