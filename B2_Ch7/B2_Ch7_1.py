# B2_Ch7_1.py

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
# from mpl_toolkits.mplot3d import axes3d
import matplotlib.tri as tri
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Delta of European option

def blsdelta(St, K, tau, r, vol, q):
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
    Delta_call  = norm.cdf(d1, loc=0, scale=1)*math.exp(-q*tau)
    Delta_put   = -norm.cdf(-d1, loc=0, scale=1)*math.exp(-q*tau)
    return Delta_call, Delta_put
    

# Initialize
tau_array = np.linspace(0.01,1,30);
St_array  = np.linspace(20,80,30);
tau_Matrix,St_Matrix = np.meshgrid(tau_array,St_array)

Delta_call_Matrix = np.empty(np.size(tau_Matrix))
Delta_put_Matrix  = np.empty(np.size(tau_Matrix))

K = 50;    # strike price
r = 0.03;  # risk-free rate
vol = 0.5; # volatility 
q = 0;     # continuously compounded yield of the underlying asset

blsdelta_vec = np.vectorize(blsdelta)
Delta_call_Matrix, Delta_put_Matrix = blsdelta_vec(St_Matrix, K, tau_Matrix, r, vol, q)

#%% plot Delta surface of European call option

plt.close('all')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_wireframe(tau_Matrix, St_Matrix, Delta_call_Matrix)

plt.show()
plt.tight_layout()
ax.set_xlabel('Time to maturity')
ax.set_ylabel('Underlying price')
ax.set_zlabel('Call Delta')

ax.xaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.yaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.zaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})

ax.set_xlim(0, 1)
ax.set_ylim(St_array.min(), St_array.max())
ax.set_zlim(Delta_call_Matrix.min(),Delta_call_Matrix.max())

#%% Call Delta surface projected to tau-Gamma

fig = plt.figure()
ax = fig.add_subplot(projection='3d') # ax = fig.gca(projection='3d')
# Or: ax = fig.add_axes(Axes3D(fig))


ax.plot_wireframe(tau_Matrix, St_Matrix, Delta_call_Matrix, color = [0.5,0.5,0.5], linewidth=0.5)

ax.contour(tau_Matrix, St_Matrix, Delta_call_Matrix, levels = 20, zdir='y', \
            offset=St_array.max(), cmap=cm.coolwarm)

ax.set_label('Call Theta')

ax.set_xlim(0, 1)
ax.set_ylim(St_array.min(), St_array.max())
ax.set_zlim(Delta_call_Matrix.min(),Delta_call_Matrix.max())

ax.xaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.yaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.zaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})

ax.set_xlabel('Time to maturity (year)')
ax.set_ylabel('Underlying price')
ax.set_zlabel('Call Delta')
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "10"

plt.tight_layout()
plt.show()

#%% Call Delta surface projected to tau-Gamma

fig = plt.figure()
ax = fig.add_subplot(projection='3d') # ax = fig.gca(projection='3d')
# Or: ax = fig.add_axes(Axes3D(fig))

ax.plot_wireframe(tau_Matrix, St_Matrix, Delta_call_Matrix, color = [0.5,0.5,0.5], linewidth=0.5)

ax.contour(tau_Matrix, St_Matrix, Delta_call_Matrix, levels = 20, zdir='x', \
           offset=0, cmap=cm.coolwarm)
# ax.contour(tau_Matrix, St_Matrix, Gamma_Matrix, levels = 20, zdir='x', \
#            cmap=cm.coolwarm)


ax.set_xlim(0, 1)
ax.set_ylim(St_array.min(), St_array.max())
ax.set_zlim(Delta_call_Matrix.min(),Delta_call_Matrix.max())

ax.xaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.yaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.zaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})

ax.set_xlabel('Time to maturity (year)')
ax.set_ylabel('Underlying price')
ax.set_zlabel('Call Delta')
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "10"

plt.tight_layout()
plt.show()

#%% Call Delta surface projected to tau-S

fig = plt.figure()
ax = fig.add_subplot(projection='3d') # ax = fig.gca(projection='3d')
# Or: ax = fig.add_axes(Axes3D(fig))

ax.plot_wireframe(tau_Matrix, St_Matrix, Delta_call_Matrix, color = [0.5,0.5,0.5], linewidth=0.5)

ax.contour(tau_Matrix, St_Matrix, Delta_call_Matrix, levels = 20, zdir='z', \
           offset=0, cmap=cm.coolwarm)

ax.set_xlim(0, 1)
ax.set_ylim(St_array.min(), St_array.max())
ax.set_zlim(Delta_call_Matrix.min(),Delta_call_Matrix.max())

ax.xaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.yaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.zaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})

ax.set_xlabel('Time to maturity (year)')
ax.set_ylabel('Underlying price')
ax.set_zlabel('Call Delta')
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "10"

plt.tight_layout()
plt.show()

#%% contour map of Call Delta

fig, ax = plt.subplots()

cntr2 = ax.contourf(tau_Matrix, St_Matrix, Delta_call_Matrix, levels = np.linspace(0,1,21), cmap="RdBu_r")

fig.colorbar(cntr2, ax=ax)

ax.contour(tau_Matrix, St_Matrix, Delta_call_Matrix, levels = [0.5], colors='k', linewidths = 2)

plt.subplots_adjust(hspace=0.5)
plt.show()
ax.set_xlabel('Time to maturity (year)')
ax.set_ylabel('Underlying price')

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "10"
#%% Compare Call vs Put Delta

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_wireframe(tau_Matrix, St_Matrix, Delta_call_Matrix)
ax.plot_wireframe(tau_Matrix, St_Matrix, Delta_put_Matrix,color = 'r')

plt.show()
plt.tight_layout()
ax.set_xlabel('Time to maturity')
ax.set_ylabel('Underlying price')
ax.set_zlabel('Delta')

ax.xaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.yaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.zaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})

ax.set_xlim(0, 1)
ax.set_ylim(St_array.min(), St_array.max())
ax.set_zlim(Delta_put_Matrix.min(),Delta_call_Matrix.max())
