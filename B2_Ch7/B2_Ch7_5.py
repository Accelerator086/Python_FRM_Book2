# B2_Ch7_5.py

###############
# Prepared by Ran An, Wei Lu, and Feng Zhang
# Editor-in-chief: Weisheng Jiang, and Sheng Tu
# Book 2  |  Financial Risk Management with Python
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2021
###############

import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import norm
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

# Gamma of European option

def blsrho(St, K, tau, r, vol, q):
    '''
    St: current price of underlying asset
    K:  strike price
    tau: time to maturity
    r: annualized risk-free rate
    vol: annualized asset price volatility
    '''
    
    d1 = (math.log(St / K) + (r - q + 0.5 * vol ** 2)\
          *tau) / (vol * math.sqrt(tau));
    d2 = d1 - vol*math.sqrt(tau);
        
    Rho_call = K*tau*math.exp(-r*tau)*norm.cdf(d2);
    Rho_put = -K*tau*math.exp(-r*tau)*norm.cdf(-d2);
    
    return Rho_call, Rho_put
    

# Initialize
tau_array = np.linspace(0.1,1,30);
St_array  = np.linspace(20,80,30);
tau_Matrix,St_Matrix = np.meshgrid(tau_array,St_array)

Vega_call_Matrix = np.empty(np.size(tau_Matrix))
Vega_put_Matrix  = np.empty(np.size(tau_Matrix))

K = 50;    # strike price
r = 0.03;  # risk-free rate
vol = 0.5; # volatility 
q = 0;     # continuously compounded yield of the underlying asset

blsrho_vec = np.vectorize(blsrho)
Rho_call_Matrix, Rho_put_Matrix = blsrho_vec(St_Matrix, K, tau_Matrix, r, vol, q)

#%% plot Rho surface of European call option

plt.close('all')

fig = plt.figure()
ax = fig.add_subplot(projection='3d') # ax = fig.gca(projection='3d')
# Or: ax = fig.add_axes(Axes3D(fig))

ax.plot_wireframe(tau_Matrix, St_Matrix, Rho_call_Matrix)
cset = ax.contour(tau_Matrix, St_Matrix, Rho_call_Matrix, zdir='z',\
                  offset=Rho_call_Matrix.min(), cmap=cm.coolwarm)
cset = ax.contour(tau_Matrix, St_Matrix, Rho_call_Matrix, zdir='x',\
                  offset=0, cmap=cm.coolwarm)
cset = ax.contour(tau_Matrix, St_Matrix, Rho_call_Matrix, zdir='y',\
                  offset=St_array.max(), cmap=cm.coolwarm)

ax.set_xlim(0, 1)
ax.set_ylim(St_array.min(), St_array.max())
ax.set_zlim(Rho_call_Matrix.min(),Rho_call_Matrix.max())


plt.tight_layout()
ax.set_xlabel('Time to maturity (year)')
ax.set_ylabel('Underlying price')
ax.set_zlabel('Call Rho')
ax.set_facecolor('white')
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "10"

plt.show()

#%% plot Rho surface of European put option

fig = plt.figure()
ax = fig.add_subplot(projection='3d') # ax = fig.gca(projection='3d')
# Or: ax = fig.add_axes(Axes3D(fig))

ax.plot_wireframe(tau_Matrix, St_Matrix, Rho_put_Matrix)
cset = ax.contour(tau_Matrix, St_Matrix, Rho_put_Matrix, zdir='z', \
                  offset=Rho_put_Matrix.min(), cmap=cm.coolwarm)
cset = ax.contour(tau_Matrix, St_Matrix, Rho_put_Matrix, zdir='x', \
                  offset=0, cmap=cm.coolwarm)
cset = ax.contour(tau_Matrix, St_Matrix, Rho_put_Matrix, zdir='y', \
                  offset=St_array.max(), cmap=cm.coolwarm)

ax.set_xlim(0, 1)
ax.set_ylim(St_array.min(), St_array.max())
ax.set_zlim(Rho_put_Matrix.min(),Rho_put_Matrix.max())

plt.tight_layout()
ax.set_xlabel('Time to maturity (year)')
ax.set_ylabel('Underlying price')
ax.set_zlabel('Put Rho')
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "10"

plt.show()
