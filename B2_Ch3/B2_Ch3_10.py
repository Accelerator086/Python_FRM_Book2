# B2_Ch3_10.py

###############
# Prepared by Ran An, Wei Lu, and Feng Zhang
# Editor-in-chief: Weisheng Jiang, and Sheng Tu
# Book 2  |  Financial Risk Management with Python
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2021
###############

B2_Ch3_10_A.py 
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import matplotlib as mpl

np.random.seed(66)

def target_dist(likelihood, prior_dist, n, k, theta):
    if theta < 0 or theta > 1:
        return 0
    else:
        return likelihood(n, theta).pmf(k)*prior_dist.pdf(theta)


likelihood = stats.binom
alpha = 20
beta = 20
prior = stats.beta(alpha, beta)
n = 100
k = 70

sigma = 0.2
theta = 0.3
accept_num = 0
MC_num = 50000

samples = np.zeros(MC_num+1)
samples[0] = theta
for i in range(MC_num):
    theta_p = theta + stats.norm(0, sigma).rvs()
    rho = min(1, target_dist(likelihood, prior, n, k, theta_p)/target_dist(likelihood, prior, n, k, theta))
    # acceptation or rejection
    u = np.random.uniform()
    if rho > u:
        accept_num += 1
        theta = theta_p
    samples[i+1] = theta

# true posterior distribution
post = stats.beta(k+alpha, n-k+beta)
thetas = np.linspace(0, 1, 200)

# assume markov chain stationary after half MC simulation number
n_stationary = len(samples)//2

# visualization
mpl.style.use('ggplot')
plt.figure(figsize=(14, 8))
plt.hist(prior.rvs(n_stationary), 50, histtype='step', density=True, linewidth=1, label='Prior distribution')
plt.hist(samples[n_stationary:], 50, histtype='step', density=True, linewidth=1, label='Target/Posterior distribution')
plt.plot(thetas, post.pdf(thetas), c='red', linestyle='--', alpha=0.5, label='True posterior distribution')
plt.xlim([0,1])
plt.legend(loc='best')



B2_Ch3_10_B.py 
# MCMC: Metropolis-Hastings algorithm
def MCMC_MH(MC_num, n, k, theta, likelihood, prior_dist, sigma):
    samples = [theta]
    while len(samples) < MC_num:
        theta_p = theta + stats.norm(0, sigma).rvs()
        rho = min(1, target_dist(likelihood, prior_dist, n, k, theta_p)/target_dist(likelihood, prior_dist, n, k, theta ))
        u = np.random.uniform()
        if rho > u:
            theta = theta_p
        samples.append(theta)
    return samples

# parameters
alpha = 20
beta = 20
prior = stats.beta(alpha, beta)
n = 100
k = 70
likelihood = stats.binom
sigma = 0.2
MC_num = 40

sample_list = [MCMC_MH(MC_num, n, k, theta, likelihood, prior, sigma) for theta in np.arange(0.1, 1, 0.2)]

# Convergence of multiple chains
for sample in sample_list:
    plt.plot(sample, '-o', markersize=8)
plt.xlim([0, MC_num])
plt.ylim([0, 1]);
plt.xlabel('Monte Carlo simulation number')
plt.ylabel('Probability')
