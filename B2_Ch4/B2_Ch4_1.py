# B2_Ch4_1.py

###############
# Prepared by Ran An, Wei Lu, and Feng Zhang
# Editor-in-chief: Weisheng Jiang, and Sheng Tu
# Book 2  |  Financial Risk Management with Python
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2021
###############

import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# define functions for data point generation
def fun1(x):
    return -2*x+3

def fun2(x):
    return 2*x+1

def fun3(x):
    return np.sin(1.5 * np.pi * x)

def fun4(x):
    return np.cos(2.1 * np.pi * (x-1.))+np.cos(3 * np.pi * x)


np.random.seed(6)

num_sample = 30

X = np.sort(np.random.rand(num_sample))

rows = 2
cols = 2
fig, axs = plt.subplots(rows, cols, figsize=(14,8))

# fig1
y1 = fun1(X) + np.random.randn(num_sample) * 0.1 # 图中的数据点，是通过函数添加“噪声”（即随机数）产生的。
polynomial_features = PolynomialFeatures(degree=1, include_bias=False)
linear_regression = LinearRegression()
pipeline = Pipeline([("polynomial_features", polynomial_features),
                     ("linear_regression", linear_regression)])
pipeline.fit(X[:, np.newaxis], y1)

X_test = np.linspace(0, 1, 1000)
axs[0, 0].plot(X_test, pipeline.predict(X_test[:, np.newaxis]), color='red', label="Fitting model")
axs[0, 0].scatter(X, y1)
axs[0, 0].set_yticks([1.0, 1.5, 2.0, 2.5, 3.0])
axs[0, 0].set_title('(a)', loc='left')

# fig2
y2 = fun2(X) + np.random.randn(num_sample) * 0.1
polynomial_features = PolynomialFeatures(degree=1, include_bias=False)
linear_regression = LinearRegression()
pipeline = Pipeline([("polynomial_features", polynomial_features),
                     ("linear_regression", linear_regression)])
pipeline.fit(X[:, np.newaxis], y2)

X_test = np.linspace(0, 1, 1000)
axs[0, 1].plot(X_test, pipeline.predict(X_test[:, np.newaxis]), color='red', label="Fitting model")
axs[0, 1].scatter(X, y2)
axs[0, 1].set_yticks([1.0, 1.5, 2.0, 2.5, 3.0])
axs[0, 1].set_title('(b)', loc='left')

# fig3
y3 = fun3(X) + np.random.randn(num_sample) * 0.1
polynomial_features = PolynomialFeatures(degree=5, include_bias=False)
linear_regression = LinearRegression()
pipeline = Pipeline([("polynomial_features", polynomial_features),
                     ("linear_regression", linear_regression)])
pipeline.fit(X[:, np.newaxis], y3)

X_test = np.linspace(0, 1, 1000)
axs[1, 0].plot(X_test, pipeline.predict(X_test[:, np.newaxis]), color='red', label="Fitting model")
axs[1, 0].scatter(X, y3)
axs[1, 0].set_title('(c)', loc='left')

# fig4
y4 = fun4(X) + np.random.randn(num_sample) * 0.1
polynomial_features = PolynomialFeatures(degree=8, include_bias=False)
linear_regression = LinearRegression()
pipeline = Pipeline([("polynomial_features", polynomial_features),
                     ("linear_regression", linear_regression)])
pipeline.fit(X[:, np.newaxis], y4)

X_test = np.linspace(0, 1, 1000)
axs[1, 1].plot(X_test, pipeline.predict(X_test[:, np.newaxis]), color='red', label="Fitting model")
axs[1, 1].scatter(X, y4)
axs[1, 1].set_yticks([-1.0, 0.0, 1.0, 2.0])
axs[1, 1].set_title('(d)', loc='left')
