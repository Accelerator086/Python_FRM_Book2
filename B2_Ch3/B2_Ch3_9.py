# B2_Ch3_9.py

###############
# Prepared by Ran An, Wei Lu, and Feng Zhang
# Editor-in-chief: Weisheng Jiang, and Sheng Tu
# Book 2  |  Financial Risk Management with Python
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2021
###############

B2_Ch3_9_A.py 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

def MC_sim_asset_price(S0, r, v, steps_per_year, T_year, MC_sim_num):
    np.random.seed(666)
    sim_steps = steps_per_year*T_year
    dt = 1/steps_per_year
    drift = (r-0.5*v*v)*dt
    voli = v*np.sqrt(dt)
    St = np.zeros(shape=(sim_steps,MC_sim_num))
    St[0,] = S0
    for i in range(1,sim_steps):
            for j in range(0,MC_sim_num):
                e = np.random.randn(1)
                St[i,j] = St[i-1,j]*np.exp(drift+voli*e)
    return St 



B2_Ch3_9_B.py 
# stock price simulation
stock_price_sim = MC_sim_asset_price(S0=100, r=0.03, v=0.3, steps_per_year=252, T_year=2, MC_sim_num=10000)

mpl.style.use('ggplot')
# show first 9 steps of simulation
plt.plot(pd.DataFrame(stock_price_sim).head(10))
plt.title("Monte Carlo simulations for stock price")
plt.xlabel("Number of simulations")
plt.ylabel("Simulated stock prices")



B2_Ch3_9_C.py 
# European call options
# discount interest rate
r = 0.03 
# strike price
K = 90 
# simulation steps per year
steps_per_year = 252 
# maturity
T_year = 2 
# monte carlo simulation number
MC_sim_num = 10000

sim_stocks = pd.DataFrame(stock_price_sim)
payoffs_eur = []

sim_steps = steps_per_year*T_year
for j in range(0, MC_sim_num):
    payoffs_eur.append(max(sim_stocks.iloc[sim_steps-1, j]-K,0)*np.exp(-r*T_year))

european_opt_price  = np.mean(payoffs_eur)

print('The price for the European call option: %.2f' % european_opt_price)



B2_Ch3_9_D.py 
# Asian call options
# discount interest rate
r = 0.03 
# strike price
K = 90 
# simulation steps per year
steps_per_year = 252 
# maturity
T_year = 2 
# monte carlo simulation number
MC_sim_num = 10000
# average time in days
ave_period = 10 

sim_stocks = pd.DataFrame(stock_price_sim)
ave_prices = []
payoffs_asian = []

sim_steps = steps_per_year*T_year
for i in range(sim_steps-ave_period, sim_steps):
    # arithmetic mean for each step
    ave_prices.append(np.mean(sim_stocks [i]))
    payoffs_asian.append(max(np.mean(sim_stocks [i])-K,0)*np.exp(-r*(i/steps_per_year)))

asian_opt_price  = np.mean(payoffs_asian)

print('The price for the Asian call option: %.2f' % asian_opt_price)
