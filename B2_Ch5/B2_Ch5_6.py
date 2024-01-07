# B2_Ch5_6.py

###############
# Prepared by Ran An, Wei Lu, and Feng Zhang
# Editor-in-chief: Weisheng Jiang, and Sheng Tu
# Book 2  |  Financial Risk Management with Python
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2021
###############

import matplotlib.pyplot as plt
import numpy as np

def Binomialtree(n, S0, K, r, vol, t, PutCall, EuropeanAmerican,Tree):  
    deltaT = t/n 
    if Tree == 'CRR':
        u = np.exp(vol*np.sqrt(deltaT))
        d = 1./u
        p = (np.exp(r*deltaT)-d) / (u-d) 
    elif Tree == 'JD':
        u = np.exp((r - vol**2*0.5)*deltaT + vol*np.sqrt(deltaT))
        d = np.exp((r - vol**2*0.5)*deltaT - vol*np.sqrt(deltaT))
        p = 0.5  
    elif Tree =='LR':
        def  h_function(z,n):
            h = 0.5+np.sign(z)*np.sqrt(0.25-0.25*np.exp(-((z/(n+1/3+0.1/(n+1)))**2)*(n+1/6)))
            return h
        
        if np.mod(n,2)>0:
            n_bar = n
        else:
            n_bar = n + 1
        
        d1 = (np.log(S0/K)+(r+vol**2/2)*t)/vol/np.sqrt(t);
        d2 = (np.log(S0/K)+(r-vol**2/2)*t)/vol/np.sqrt(t);
        pbar = h_function(d1,n_bar)
        p = h_function(d2,n_bar)
        u = np.exp(r*deltaT)*pbar/p
        d = (np.exp(r*deltaT)-p*u)/(1-p)  
            
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
    
    scatter_x_stock = [0.0]
    scatter_y_stock = [stockvalue[0,0]]
    
    plt.figure(1)
    for i in range(1,n+1):
        for j in range(1,i+1):
            
            x_stock_tree_u = [(i-1)*deltaT]
            x_stock_tree_d = [(i-1)*deltaT]
            y_stock_tree_upper = [stockvalue[i-1,j-1]]
            y_stock_tree_lower = [stockvalue[i-1,j-1]]
                         
            x_temp = i*deltaT 
            y_temp1 = stockvalue[i,j-1]
            y_temp3 = stockvalue[i,j]
            
            x_stock_tree_u.append(x_temp)
            x_stock_tree_d.append(x_temp)
            scatter_x_stock.append(i*deltaT)
                        
            y_stock_tree_lower.append(y_temp1)
            y_stock_tree_upper.append(y_temp3)
            scatter_y_stock.append(stockvalue[i,j-1])
                      
            if Tree == 'CRR':
                plt.plot(np.array(x_stock_tree_u), np.array(y_stock_tree_upper),'b-o',linewidth=0.4,markersize = 2)
                plt.plot(np.array(x_stock_tree_d), np.array(y_stock_tree_lower),'b-o',linewidth=0.4,markersize = 2)
            elif Tree == 'JD':
                plt.plot(np.array(x_stock_tree_u), np.array(y_stock_tree_upper),'r-o',linewidth=0.4,markersize = 2)
                plt.plot(np.array(x_stock_tree_d), np.array(y_stock_tree_lower),'r-o',linewidth=0.4,markersize = 2)
            elif Tree == 'LR':
                plt.plot(np.array(x_stock_tree_u), np.array(y_stock_tree_upper),'r-o',linewidth=0.4,markersize = 2)
                plt.plot(np.array(x_stock_tree_d), np.array(y_stock_tree_lower),'r-o',linewidth=0.4,markersize = 2)    
            else:
                print("Tree type not supported")  
    
    plt.xlabel('Time (year)',fontsize=8)
    plt.ylabel('Underlying price',fontsize=8)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)    

    plt.figure(2)
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
                             
            y_option_tree_upper.append(y_temp1)
            y_option_tree_lower.append(y_temp3)
            
            if Tree == 'CRR':
                plt.plot(np.array(x_option_tree_u), np.array(y_option_tree_upper),'b-o',linewidth=0.5,markersize = 2)
                plt.plot(np.array(x_option_tree_d), np.array(y_option_tree_lower),'b-o',linewidth=0.5,markersize = 2)
            elif Tree == 'JD':
                plt.plot(np.array(x_option_tree_u), np.array(y_option_tree_upper),'r-o',linewidth=0.5,markersize = 2)
                plt.plot(np.array(x_option_tree_d), np.array(y_option_tree_lower),'r-o',linewidth=0.5,markersize = 2)
            elif Tree == 'LR':
                plt.plot(np.array(x_option_tree_u), np.array(y_option_tree_upper),'r-o',linewidth=0.5,markersize = 2)
                plt.plot(np.array(x_option_tree_d), np.array(y_option_tree_lower),'r-o',linewidth=0.5,markersize = 2)    
            else:
                print("Tree type not supported") 
    
    plt.xlabel('Time (year)',fontsize=8)
    if PutCall=="Put":
        plt.ylabel('Put option price',fontsize=8)    
    else:
        plt.ylabel('Call option price',fontsize=8)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)             
           
    return optionvalue[0,0]

    # Inputs
n = 20    #number of steps
S0 = 50  #initial underlying asset price
r = 0.01  #risk-free interest rate
K = 55   #strike price
vol = 0.3 #volatility
t = 1.0

y1 = Binomialtree(n, S0, K, r, vol, t, PutCall="Call", EuropeanAmerican="American",Tree = 'CRR')
y2 = Binomialtree(n, S0, K, r, vol, t, PutCall="Call", EuropeanAmerican="American",Tree = 'JD')
