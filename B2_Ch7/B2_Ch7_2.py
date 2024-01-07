# B2_Ch7_2.py

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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Gamma of European option

def blsgamma(St, K, tau, r, vol, q):
    '''
    St: current price of underlying asset
    K:  strike price
    tau: time to maturity
    r: annualized risk-free rate
    vol: annualized asset price volatility
    '''
    
    d1 = (math.log(St / K) + (r - q + 0.5 * vol ** 2)\
          *tau) / (vol * math.sqrt(tau));
        
    Gamma = math.exp(-q*tau)*norm.pdf(d1)/St/vol/math.sqrt(tau);

    return Gamma
    

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

blsgamma_vec = np.vectorize(blsgamma)
Gamma_Matrix = blsgamma_vec(St_Matrix, K, tau_Matrix, r, vol, q)

#%% plot Gamma surface of European call option

plt.close('all')

# Normalize to [0,1]
norm = plt.Normalize(Gamma_Matrix.min(), Gamma_Matrix.max())
colors = cm.coolwarm(norm(Gamma_Matrix))

fig = plt.figure()
ax = fig.add_subplot(projection='3d') # ax = fig.gca(projection='3d')
# Or: ax = fig.add_axes(Axes3D(fig))
surf = ax.plot_surface(tau_Matrix, St_Matrix, Gamma_Matrix,
    facecolors=colors, shade=False)
surf.set_facecolor((0,0,0,0))
plt.show()

plt.tight_layout()
ax.set_xlabel('Time to maturity')
ax.set_ylabel('Underlying price')
ax.set_zlabel('Gamma')

ax.set_xlim(0, 1)
ax.set_ylim(St_array.min(), St_array.max())
ax.set_zlim(Gamma_Matrix.min(),Gamma_Matrix.max())

ax.xaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.yaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.zaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "10"

#%% Gamma surface projected to S-Gamma

fig = plt.figure()
ax = fig.add_subplot(projection='3d') # ax = fig.gca(projection='3d')
# Or: ax = fig.add_axes(Axes3D(fig))

ax.plot_wireframe(tau_Matrix, St_Matrix, Gamma_Matrix, color = [0.5,0.5,0.5], linewidth=0.5)

ax.contour(tau_Matrix, St_Matrix, Gamma_Matrix, levels = 20, zdir='x', \
           offset=0, cmap=cm.coolwarm)
# ax.contour(tau_Matrix, St_Matrix, Gamma_Matrix, levels = 20, zdir='x', \
#            cmap=cm.coolwarm)
    
# cbar = fig.colorbar(csetf, ax=ax,orientation='horizontal')
# cbar.set_label('Call Gamma')

ax.set_xlim(0, 1)
ax.set_ylim(St_array.min(), St_array.max())
ax.set_zlim(Gamma_Matrix.min(),Gamma_Matrix.max())

ax.xaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.yaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.zaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})

ax.set_xlabel('Time to maturity (year)')
ax.set_ylabel('Underlying price')
ax.set_zlabel('Gamma')
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "10"

plt.tight_layout()
plt.show()

#%% Gamma surface projected to tau-Gamma

fig = plt.figure()
ax = fig.add_subplot(projection='3d') # ax = fig.gca(projection='3d')
# Or: ax = fig.add_axes(Axes3D(fig))

ax.plot_wireframe(tau_Matrix, St_Matrix, Gamma_Matrix, color = [0.5,0.5,0.5], linewidth=0.5)

ax.contour(tau_Matrix, St_Matrix, Gamma_Matrix, levels = 20, zdir='y', \
            offset=St_array.max(), cmap=cm.coolwarm)

# cbar = fig.colorbar(csetf, ax=ax,orientation='horizontal')
# cbar.set_label('Call Gamma')

ax.set_xlim(0, 1)
ax.set_ylim(St_array.min(), St_array.max())
ax.set_zlim(Gamma_Matrix.min(),Gamma_Matrix.max())

ax.xaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.yaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.zaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})

ax.set_xlabel('Time to maturity (year)')
ax.set_ylabel('Underlying price')
ax.set_zlabel('Gamma')
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "10"

plt.tight_layout()
plt.show()

#%% Gamma surface projected to tau-S

fig = plt.figure()
ax = fig.add_subplot(projection='3d') # ax = fig.gca(projection='3d')
# Or: ax = fig.add_axes(Axes3D(fig))

ax.plot_wireframe(tau_Matrix, St_Matrix, Gamma_Matrix, color = [0.5,0.5,0.5], linewidth=0.5)

ax.contour(tau_Matrix, St_Matrix, Gamma_Matrix, levels = 20, zdir='z', \
            offset=Gamma_Matrix.min(), cmap=cm.coolwarm)

# cbar = fig.colorbar(csetf, ax=ax,orientation='horizontal')

ax.set_xlim(0, 1)
ax.set_ylim(St_array.min(), St_array.max())
ax.set_zlim(Gamma_Matrix.min(),Gamma_Matrix.max())

ax.xaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.yaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.zaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})

ax.set_xlabel('Time to maturity (year)')
ax.set_ylabel('Underlying price')
ax.set_zlabel('Gamma')
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "10"

plt.tight_layout()
plt.show()

#%% contour map

fig, ax = plt.subplots()

cntr2 = ax.contourf(tau_Matrix, St_Matrix, Gamma_Matrix, levels = 20, cmap="RdBu_r")

fig.colorbar(cntr2, ax=ax)

plt.show()

ax.set_xlabel('Time to maturity')
ax.set_ylabel('Underlying price')

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "10"
