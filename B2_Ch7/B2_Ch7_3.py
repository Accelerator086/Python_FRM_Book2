# B2_Ch7_3.py

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
import matplotlib.tri as tri
from matplotlib import cm

# Delta of European option

def blstheta(St, K, tau, r, vol, q):
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
    
    Theta_call = -math.exp(-q*tau)*St*norm.pdf(d1)*vol/2/math.sqrt(tau) - \
        r*K*math.exp(-r*tau)*norm.cdf(d2) + q*St*math.exp(-q*tau)*norm.cdf(d1)
        
    Theta_put = -math.exp(-q*tau)*St*norm.pdf(-d1)*vol/2/math.sqrt(tau) + \
        r*K*math.exp(-r*tau)*norm.cdf(-d2) - q*St*math.exp(-q*tau)*norm.cdf(-d1)
    return Theta_call, Theta_put

# Initialize
tau_array = np.linspace(0.05,1,30);
St_array  = np.linspace(20,80,30);
tau_Matrix,St_Matrix = np.meshgrid(tau_array,St_array)

Theta_call_Matrix = np.empty(np.size(tau_Matrix))
Theta_put_Matrix  = np.empty(np.size(tau_Matrix))

K = 50;    # strike price
r = 0.03;  # risk-free rate
vol = 0.5; # volatility 
q = 0;     # continuously compounded yield of the underlying asset

blstheta_vec = np.vectorize(blstheta)
Theta_call_Matrix, Theta_put_Matrix = blstheta_vec(St_Matrix, K, tau_Matrix, r, vol, q)

#%% plot Theta surface of European call option

plt.close('all')

fig = plt.figure()
ax = fig.add_subplot(projection='3d') # ax = fig.gca(projection='3d')
# Or: ax = fig.add_axes(Axes3D(fig))

ax.plot_wireframe(tau_Matrix, St_Matrix, Theta_call_Matrix, color = [0.5,0.5,0.5], linewidth=0.5)
csetf = ax.contourf(tau_Matrix, St_Matrix, Theta_call_Matrix, levels = 15, zdir='z', \
                    offset=Theta_call_Matrix.min(), cmap=cm.coolwarm)
ax.contour(tau_Matrix, St_Matrix, Theta_call_Matrix, levels = 15, zdir='x', \
           offset=0, cmap=cm.coolwarm)
ax.contour(tau_Matrix, St_Matrix, Theta_call_Matrix, levels = 15, zdir='y', \
           offset=St_array.max(), cmap=cm.coolwarm)

cbar = fig.colorbar(csetf, ax=ax,orientation='horizontal')
cbar.set_label('Call Theta')

ax.set_xlim(0, 1)
ax.set_ylim(St_array.min(), St_array.max())
ax.set_zlim(Theta_call_Matrix.min(),Theta_call_Matrix.max())

ax.xaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.yaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.zaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})

ax.set_xlabel('Time to maturity (year)')
ax.set_ylabel('Underlying price')
ax.set_zlabel('Call Theta')
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "10"

plt.tight_layout()
plt.show()


#%% plot Theta surface of European put option


fig = plt.figure()
ax = fig.add_subplot(projection='3d') # ax = fig.gca(projection='3d')
# Or: ax = fig.add_axes(Axes3D(fig))

ax.plot_wireframe(tau_Matrix, St_Matrix, Theta_put_Matrix, color = [0.5,0.5,0.5], linewidth=0.5)
csetf = ax.contourf(tau_Matrix, St_Matrix, Theta_put_Matrix, levels = 15, zdir='z', \
                    offset=Theta_put_Matrix.min(), cmap=cm.coolwarm)
ax.contour(tau_Matrix, St_Matrix, Theta_put_Matrix, levels = 15, zdir='x', \
           offset=0, cmap=cm.coolwarm)
ax.contour(tau_Matrix, St_Matrix, Theta_put_Matrix, levels = 15, zdir='y', \
           offset=St_array.max(), cmap=cm.coolwarm)

cbar = fig.colorbar(csetf, ax=ax, orientation='horizontal')
# cbar.set_label('Put Theta')

ax.set_xlim(0, 1)
ax.set_ylim(St_array.min(), St_array.max())
ax.set_zlim(Theta_put_Matrix.min(),Theta_put_Matrix.max())

ax.xaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.yaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.zaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})

ax.set_xlabel('Time to maturity (year)')
ax.set_ylabel('Underlying price')
ax.set_zlabel('Put Theta')
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "10"

plt.tight_layout()
plt.show()

fig, ax = plt.subplots()

cntr2 = ax.contourf(tau_Matrix, St_Matrix, Theta_put_Matrix, levels = 20, cmap="RdBu_r")
ax.contour(tau_Matrix, St_Matrix, Theta_put_Matrix, levels = 0,colors='k', linewidths = 2)

fig.colorbar(cntr2, ax=ax)
plt.show()

ax.set_xlabel('Time to maturity (year)')
ax.set_ylabel('Underlying price')

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "10"
ax.set_ylim(St_array.min(), St_array.max())

#%% Compare Call vs Put Theta

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_wireframe(tau_Matrix, St_Matrix, Theta_call_Matrix)
ax.plot_wireframe(tau_Matrix, St_Matrix, Theta_put_Matrix,color = 'r')

plt.show()
plt.tight_layout()
ax.set_xlabel('Time to maturity')
ax.set_ylabel('Underlying price')
ax.set_zlabel('Theta')

ax.xaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.yaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.zaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})

ax.set_xlim(0, 1)
ax.set_ylim(St_array.min(), St_array.max())
ax.set_zlim(Theta_call_Matrix.min(),Theta_put_Matrix.max())

