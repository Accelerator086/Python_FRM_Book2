# B2_Ch4_2.py

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

def original_fun(X):
    return np.sin(1.5 * np.pi * X)

np.random.seed(6)

num_sample = 30
degrees = [1, 5, 15]
titles = ['(a) Underfitting', '(b) Optimalfitting', '(c) Overfitting']
X = np.sort(np.random.rand(num_sample))
y = original_fun(X) + np.random.randn(num_sample) * 0.1

rows = 1
cols = 3
fig, axs = plt.subplots(rows, cols, figsize=(14,5))

for i in range(len(degrees)):
    polynomial_features = PolynomialFeatures(degree=degrees[i],
                                             include_bias=False)
    linear_regression = LinearRegression()
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])
    pipeline.fit(X[:, np.newaxis], y)

    X_test = np.linspace(0, 1, 100)
    axs[i].plot(X_test, pipeline.predict(X_test[:, np.newaxis]), color='red', label="Fitting model")
    axs[i].plot(X_test, original_fun(X_test), color='lightblue', label="Original function")
    axs[i].scatter(X, y, s=20, label="Samples")
    axs[i].set_xlim(0, 1)
    axs[i].set_ylim(-2, 2)
    axs[i].set_xticks([0.0, 0.5, 1.0])
    axs[i].set_yticks([-2, -1, 0, 1, 2])
    axs[i].legend(loc="best")
    axs[i].set_title(titles[i], loc='left')
