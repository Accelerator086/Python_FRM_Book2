# B2_Ch3_4.py

###############
# Prepared by Ran An, Wei Lu, and Feng Zhang
# Editor-in-chief: Weisheng Jiang, and Sheng Tu
# Book 2  |  Financial Risk Management with Python
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2021
###############

B2_Ch3_4_A.py 
import mcint 
import random 
import math 
import matplotlib.pyplot as plt

def w(r, theta, phi, alpha, beta, gamma): 
    return(-math.log(theta * beta)) 

def integrand(x): 
    r  = x[0] 
    theta = x[1] 
    alpha = x[2] 
    beta = x[3] 
    gamma = x[4] 
    phi = x[5] 
    k = 1. 
    T = 1. 
    ww = w(r, theta, phi, alpha, beta, gamma) 
    return (math.exp(-ww/(k*T)) - 1.)*r*r*math.sin(beta)*math.sin(theta) 

def sampler(): 
    while True: 
     r  = random.uniform(0.,1.) 
     theta = random.uniform(0.,2.*math.pi) 
     alpha = random.uniform(0.,2.*math.pi) 
     beta = random.uniform(0.,2.*math.pi) 
     gamma = random.uniform(0.,2.*math.pi) 
     phi = random.uniform(0.,math.pi) 
     yield (r, theta, alpha, beta, gamma, phi) 

domainsize = math.pow(2*math.pi,4)*math.pi*1 
expected = 16*math.pow(math.pi,5)/3. 

MC_num_list = [50000, 100000, 500000, 1000000, 5000000, 10000000, 50000000, 100000000]
relative_error_list = []
for MC_num in MC_num_list: 
    random.seed(1) 
    # monte carlo integration via mcint library
    result, error = mcint.integrate(integrand, sampler(), measure=domainsize, n=MC_num) 
    diff = abs(result - expected) 
    relative_error = diff/expected
    relative_error_list.append(relative_error)
    print("Monte Carlo simulation number: ", MC_num) 
    print("Monte Carlo simulation result: ", round(result,2), " estimated error: ", round(error,2)) 
    print ("True result = ", round(expected,2))
    print ("Relative error: {:.2%}".format(relative_error))


B2_Ch3_4_B.py 
plt.plot(MC_num_list, relative_error_list, 'ro')
plt.xscale('log')
plt.xlabel('Monte Carlo simulation number')
plt.ylabel('Relative error')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().yaxis.set_ticks_position('left')
plt.gca().xaxis.set_ticks_position('bottom')
