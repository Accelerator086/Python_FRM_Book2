# B2_Ch4_10.py

###############
# Prepared by Ran An, Wei Lu, and Feng Zhang
# Editor-in-chief: Weisheng Jiang, and Sheng Tu
# Book 2  |  Financial Risk Management with Python
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2021
###############

B2_Ch4_10_A.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

# create lasso regression fit and plot function
def lasso_regression_fit_plot(data, predictors, alpha, alpha_subplotpos):
    # fit lasso regression model
    lassoregrmodel = Lasso(alpha=alpha, normalize=True, tol=0.1)
    lassoregrmodel.fit(data[predictors], data['y'])
    y_pred = lassoregrmodel.predict(data[predictors])
    
    # plot for model with predefined alpha
    if alpha in alpha_subplotpos:
        plt.subplot(alpha_subplotpos[alpha])
        plt.plot(data['x'], data['y'],'.')
        plt.plot(data['x'], y_pred, 'r')         
        plt.title('$\\alpha$=%.3g'%alpha)
    plt.yticks([-1.0, -0.5, 0, 0.5, 1.0])
    
    # return results
    rss = sum((y_pred-data['y'])**2)
    ret = [rss]
    ret.extend([lassoregrmodel.intercept_])
    ret.extend(lassoregrmodel.coef_)
    return ret


B2_Ch4_10_B.py
# extract raw data
data = pd.read_csv(r'.\RidgeRegrData.csv')

# prepare data with powers up to 15
for i in range(2,16):  
    colname = 'x_%d'%i      
    data[colname] = data['x']**i

# initialize predictors to be set of 15 powers of x
predictors=['x']
predictors.extend(['x_%d'%i for i in range(2,16)])

# set list of alpha values
alpha_list = [1e-20, 1e-10, 1e-5, 1e-3, 1e-2, 1e-1, 1, 2, 3, 5, 10, 20]

# store coefficients
col = ['rss','intercept'] + ['coef_x_%d'%i for i in range(1,16)]
ind = ['alpha_%.2g'%alpha_list[i] for i in range(0,len(alpha_list))]
coef_matrix_lasso = pd.DataFrame(index=ind, columns=col)

# alpha:subplot position
alpha_subplotpos = {1e-20:231, 1e-10:232, 1e-5:233, 1e-3:234, 1e-2:235, 1e-1:236}
for i in range(len(alpha_list)):
    coef_matrix_lasso.iloc[i,] = lasso_regression_fit_plot(data, predictors, alpha_list[i], alpha_subplotpos)


B2_Ch4_10_C.py
# show parameter matrix
pd.options.display.float_format = '{:,.2g}'.format
coef_matrix_lasso


B2_Ch4_10_D.py
# plot rss of models
plt.plot(coef_matrix_lasso['rss'], 'o')
plt.title('RSS Trend')
plt.xlabel(r'$\alpha$')
plt.xticks(rotation=30)
plt.ylabel('RSS')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().yaxis.set_ticks_position('left')
plt.gca().xaxis.set_ticks_position('bottom')


B2_Ch4_10_E.py
coef_matrix_lasso.apply(lambda x: sum(x.values==0),axis=1)
