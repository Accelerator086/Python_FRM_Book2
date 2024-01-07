# B2_Ch7_4.py

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

# Vega of European option

def blsvega(St, K, tau, r, vol, q):
    '''
    St: current price of underlying asset
    K:  strike price
    tau: time to maturity
    r: annualized risk-free rate
    vol: annualized asset price volatility
    '''
    
    d1 = (math.log(St / K) + (r - q + 0.5 * vol ** 2)\
          *tau) / (vol * math.sqrt(tau));
    
    Vega = St*math.exp(-q*tau)*norm.pdf(d1)*math.sqrt(tau)
    return Vega

# Initialize
tau_array = np.linspace(0.1,1,30);
St_array  = np.linspace(20,80,30);
tau_Matrix,St_Matrix = np.meshgrid(tau_array,St_array)

Delta_call_Matrix = np.empty(np.size(tau_Matrix))
Delta_put_Matrix  = np.empty(np.size(tau_Matrix))

K = 50;    # strike price
r = 0.03;  # risk-free rate
vol = 0.5; # volatility 
q = 0;     # continuously compounded yield of the underlying asset

blsvega_vec = np.vectorize(blsvega)
Vega_Matrix = blsvega_vec(St_Matrix, K, tau_Matrix, r, vol, q)

#%% plot Vega surface of European call/put option

plt.close('all')

# Normalize to [0,1]
norm = plt.Normalize(Vega_Matrix.min(), Vega_Matrix.max())
colors = cm.coolwarm(norm(Vega_Matrix))

fig = plt.figure()
ax = fig.add_subplot(projection='3d') # ax = fig.gca(projection='3d')
# Or: ax = fig.add_axes(Axes3D(fig))
surf = ax.plot_surface(tau_Matrix, St_Matrix, Vega_Matrix,
    facecolors=colors, shade=False)
surf.set_facecolor((0,0,0,0))
plt.show()

plt.tight_layout()
ax.set_xlabel('Time to maturity')
ax.set_ylabel('Underlying price')
ax.set_zlabel('Call/Put Vega')

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "10"
ax.set_xlim(0, 1)
ax.set_ylim(St_array.min(), St_array.max())


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y, Z = axes3d.get_test_data(0.05)
cset = ax.contour(tau_Matrix, St_Matrix, Vega_Matrix, cmap=cm.coolwarm,levels = 20)
ax.clabel(cset, fontsize=9, inline=1)

plt.show()
plt.tight_layout()
ax.set_xlabel('Time to maturity')
ax.set_ylabel('Underlying price')
ax.set_zlabel('Call/Put Vega')

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "10"
ax.set_xlim(0, 1)
ax.set_ylim(St_array.min(), St_array.max())

# contour map

fig, ax = plt.subplots()

cntr2 = ax.contourf(tau_Matrix, St_Matrix, Vega_Matrix, levels = 20, cmap="RdBu_r")

fig.colorbar(cntr2, ax=ax)
# ax.set(xlim=(-2, 2), ylim=(-2, 2))
# plt.subplots_adjust(hspace=0.5)
plt.show()

ax.set_xlabel('Time to maturity (year)')
ax.set_ylabel('Underlying price')

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "10"
ax.set_ylim(St_array.min(), St_array.max())
