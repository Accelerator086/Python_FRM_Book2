# B2_Ch4_8.py

###############
# Prepared by Ran An, Wei Lu, and Feng Zhang
# Editor-in-chief: Weisheng Jiang, and Sheng Tu
# Book 2  |  Financial Risk Management with Python
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2021
###############

import pandas as pd
from scipy import stats 
import matplotlib.pyplot as plt

df = pd.read_csv(r'.\outliersimpact.csv')

X = df.x
y = df.y

plt.plot(X, y, 'bo')

slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(X, y)
rline1 = intercept1 + slope1*X

plt.plot(X, rline1,'r-', label='Fitting with outliers')

plt.annotate('Fitting with outliers', xy=(0.6, intercept1 + slope1*0.6), xytext=(0.6, 1.2), 
              arrowprops=dict(arrowstyle="-|>",
                             connectionstyle="arc3",
                             mutation_scale=20,
                             fc="w"))

plt.annotate('outliers', xy=(0.802171, 0.5), xytext=(0.75, 0.6))#, 
             # arrowprops=dict(arrowstyle="-|>",
             #                connectionstyle="arc3",
             #                mutation_scale=20,
             #                fc="w"))
plt.annotate('', xy=(0.89286, 0.6), xytext=(0.75, 0.6))#, 
                          # arrowprops=dict(arrowstyle="-|>",
                          #               connectionstyle="arc3",
                          #               mutation_scale=20,
                          #               fc="w"))

# eliminate two outliers
df_nooutliers = df[(df['y']!=0.5) & (df['y']!=0.6)]

X = df_nooutliers.x
y = df_nooutliers.y

slope2, intercept2, r_value2, p_valu2e, std_err2 = stats.linregress(X, y)
rline2 = intercept2 + slope2*X
plt.plot(X, rline2,'r--', label='Fitting without outliers')

plt.annotate('Fitting without outliers', xy=(0.7, intercept2 + slope2*0.7), xytext=(0.4, 2.2), 
              arrowprops=dict(arrowstyle="-|>",
                             connectionstyle="arc3",
                             mutation_scale=20,
                             fc="w"))

plt.title('Impact on linear regression by outliers')
plt.gca().set_yticks([0.5, 1.0, 1.5, 2.0, 2.5])

plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().yaxis.set_ticks_position('left')
plt.gca().xaxis.set_ticks_position('bottom')
