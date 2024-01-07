# B2_Ch4_9.py

###############
# Prepared by Ran An, Wei Lu, and Feng Zhang
# Editor-in-chief: Weisheng Jiang, and Sheng Tu
# Book 2  |  Financial Risk Management with Python
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2021
###############

B2_Ch4_9_A.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import Normalizer,StandardScaler,MinMaxScaler,MaxAbsScaler,RobustScaler
from sklearn.pipeline import make_pipeline


B2_Ch4_9_B.py
# extract and plot raw data
data = pd.read_csv(r'.\RidgeRegrData.csv')
plt.plot(data['x'], data['y'], 'o')
plt.title('Raw Data')
plt.xlabel('x')
plt.ylabel('y')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().yaxis.set_ticks_position('left')
plt.gca().xaxis.set_ticks_position('bottom')


B2_Ch4_9_C.py
# prepare data with powers up to 15
for i in range(2,16):  
    colname = 'x_%d'%i      
    data[colname] = data['x']**i
print(data.head())


B2_Ch4_9_D.py
# create ridge regression fit and plot function
def ridge_regression_fit_plot(data, predictors, alpha, alpha_subplotpos):
    # fit ridge regression model
    ridgeregrmodel = make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=alpha*data[predictors].shape[0])) # Ridge(alpha=alpha, normalize=True)  ,with_std=False
        # https://scikit-learn.org/1.0/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge
        # Deprecated since version 1.0: normalize was deprecated in version 1.0 and will be removed in 1.2.
    # kwargs = {s[0] + '__sample_weight': sample_weight for s in ridgeregrmodel.steps}
    ridgeregrmodel.fit(data[predictors], data['y'] ) # **kwargs
    y_pred = ridgeregrmodel.predict(data[predictors])
    
    # plot for model with predefined alpha
    if alpha in alpha_subplotpos:
        plt.subplot(alpha_subplotpos[alpha])
        plt.tight_layout()
        plt.plot(data['x'], data['y'],'.')
        plt.plot(data['x'], y_pred, 'g-')         
        plt.title('Ridge Regression:  $\\alpha$=%.3g'%alpha)
    
    # return results
    rss = sum((y_pred-data['y'])**2)
    ret = [rss]
    ret.extend([ridgeregrmodel.named_steps['ridge'].intercept_]) # ;print(ridgeregrmodel.named_steps['ridge'].intercept_)
    ret.extend(ridgeregrmodel.named_steps['ridge'].coef_/ridgeregrmodel.named_steps['standardscaler'].scale_) # model["standardscaler"].transform(model["ridgecv"].coef_.reshape(1, -1))
    return ret


B2_Ch4_9_E.py
# initialize predictors to be set of 15 powers of x
predictors=['x']
predictors.extend(['x_%d'%i for i in range(2,16)])
# set list of alpha values
alpha_list = [1e-20, 1e-10, 1e-5, 1e-3, 1e-2, 1e-1, 1, 2, 3, 5, 10, 20]
# store coefficients
col = ['rss','intercept'] + ['coef_x_%d'%i for i in range(1,16)]
ind = ['alpha_%.2g' % alpha_list[i] for i in range(0,len(alpha_list))]
coef_matrix_ridge = pd.DataFrame(index=ind, columns=col)
# alpha:subplot position
alpha_subplotpos = {1e-20:241, 1e-10:242, 1e-3:243, 1e-2:244, 1e-1:245, 1:246, 5:247, 20:248}
for i in range(len(alpha_list)):
    coef_matrix_ridge.iloc[i,] = ridge_regression_fit_plot(data, predictors, alpha_list[i], alpha_subplotpos)  

# # 
# ridgeregrmodel = make_pipeline(StandardScaler(), Ridge(alpha=1e-20))
# ridgeregrmodel.fit(data[predictors], data['y'])
# ridgeregrmodel['ridge'].intercept_ # 10451.295451948972,22048.293021698686
# ridgeregrmodel.named_steps['ridge'].coef_
# # array([-7.20560278e+04,  1.16289717e+06, -9.27885090e+06,  4.67465778e+07,
# #        -1.61059002e+08,  3.89294885e+08, -6.52028102e+08,  6.99697019e+08,
# #        -3.20658078e+08, -3.21176145e+08,  7.46921025e+08, -6.90473148e+08,
# #         3.64908604e+08, -1.07945895e+08,  1.39602736e+07])

B2_Ch4_9_F.py
# show parameter matrix
pd.options.display.float_format = '{:,.2g}'.format
coef_matrix_ridge

B2_Ch4_9_G.py
# plot rss of models
plt.plot(coef_matrix_ridge['rss'], 'o')
plt.title('RSS Trend')
plt.xlabel(r'$\alpha$')
plt.xticks(rotation=30)
plt.ylabel('RSS')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().yaxis.set_ticks_position('left')
plt.gca().xaxis.set_ticks_position('bottom')


B2_Ch4_9_H.py
coef_matrix_ridge.apply(lambda x: sum(x.values==0),axis=1)

