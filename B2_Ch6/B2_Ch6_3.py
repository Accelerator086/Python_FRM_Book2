# B2_Ch6_3.py

###############
# Prepared by Ran An, Wei Lu, and Feng Zhang
# Editor-in-chief: Weisheng Jiang, and Sheng Tu
# Book 2  |  Financial Risk Management with Python
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2021
###############

import matplotlib.pyplot as plt
import numpy as np

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

    # Inputs
n = 50
r = 0.085  #risk-free interest rate
q = 0.0 # dividend yield
K = 55   #strike price
vol = 0.45 #volatility
t_base = 2.0
PutCall = 1 # 1 for call;-1 for put
spot = np.arange(0,100,1)  #initial underlying asset price

t= np.arange(0.00001, 2, 0.25)
price_call = np.zeros(len(spot)) 
price_put = np.zeros(len(spot)) 

NUM_COLORS = len(t)
cm = plt.get_cmap('RdYlBu')
fig1 = plt.figure(1)
ax = fig1.add_subplot(111)

for i in range(len(t)):
    t_tmp = t[i]
    price_put = np.array([Binomialtree(n, S0, K, r, q, vol, t_tmp, PutCall="Put", EuropeanAmerican="American") for S0 in spot])
    lines = ax.plot(spot, price_put, label='price')
    lines[0].set_color(cm(i/NUM_COLORS))

plt.xlabel("Stock price, S (USD)")
plt.ylabel("Am. put option price, P (USD)")
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.grid(linestyle='--', axis='both', linewidth=0.25, color=[0.5,0.5,0.5])
plt.gca().legend(['T = 0.00 Yr','T = 0.25 Yr','T = 0.50 Yr','T = 0.75 Yr','T = 1.00 Yr','T = 1.25 Yr','T = 1.50 Yr','T = 1.75 Yr'],loc='upper right')