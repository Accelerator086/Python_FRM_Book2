# B2_Ch5_4.py

###############
# Prepared by Ran An, Wei Lu, and Feng Zhang
# Editor-in-chief: Weisheng Jiang, and Sheng Tu
# Book 2  |  Financial Risk Management with Python
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2021
###############

import matplotlib.pyplot as plt
import numpy as np
import scipy.special

def Binomialtree(n, S0, K, r, vol, t, PutCall, EuropeanAmerican):  
    deltaT = t/n 
    u = np.exp(vol*np.sqrt(deltaT))
    d = 1./u
    p = (np.exp(r*deltaT)-d) / (u-d) 
    
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
       
    scatter_x_option = [0.0]
    scatter_y_option = [optionvalue[0,0]]
    scatter_prob_option = [1.0]     
    plt.figure()        
            
    for i in range(1,n+1):
        for j in range(1,i+1):
            x_option_tree_u = [(i-1)*deltaT]
            x_option_tree_d = [(i-1)*deltaT]
            y_option_tree_upper = [optionvalue[i-1,j-1]]
            y_option_tree_lower = [optionvalue[i-1,j-1]]
                         
            x_temp = i*deltaT
            
            y_temp1 = optionvalue[i,j-1]
            y_temp3 = optionvalue[i,j]
            
            x_option_tree_u.append(x_temp)
            x_option_tree_d.append(x_temp)
            scatter_x_option.append(i*deltaT)
                        
            y_option_tree_upper.append(y_temp1)
            y_option_tree_lower.append(y_temp3)
            scatter_y_option.append(optionvalue[i,j-1])
            temp_prob = scipy.special.comb(i, j - 1, exact=True)*p**(j - 1)*(1 - p)**(i-j+1)

            scatter_prob_option.append(temp_prob)
            
            plt.plot(np.array(x_option_tree_u), np.array(y_option_tree_upper),'b-',linewidth=0.5)
            plt.plot(np.array(x_option_tree_d), np.array(y_option_tree_lower),'b-',linewidth=0.5)
        
        temp_prob = scipy.special.comb(i, j, exact=True)*p**(j)*(1 - p)**(i-j)
        scatter_prob_option.append(temp_prob)    
        scatter_x_option.append(i*deltaT)
        scatter_y_option.append(optionvalue[i,j])
    
    colors = scatter_prob_option
    plt.scatter(np.array(scatter_x_option),np.array(scatter_y_option),c=colors,alpha=0.5,cmap ='RdBu_r') 
    plt.xlabel('Time (year)',fontsize=8)
    if  PutCall=="Put":
        plt.ylabel('Put option price',fontsize=8)
    else: 
        plt.ylabel('Call option price',fontsize=8)
        
    # plt.gca().set_aspect(1)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.colorbar()
           
    return optionvalue[0,0]
    # Inputs
n = 20    #number of steps
S0 = 50  #initial underlying asset price
r = 0.01  #risk-free interest rate
K = 55   #strike price
vol = 0.3 #volatility
t = 1.0
y = Binomialtree(n, S0, K, r, vol, t, PutCall="Call", EuropeanAmerican="American")
y = Binomialtree(n, S0, K, r, vol, t, PutCall="Put", EuropeanAmerican="American")