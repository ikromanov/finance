#
# Monte Carlo valuation of Asian call option
# in Black-Scholes-Merton model
# bsm_mcs_asian.py
#
import numpy as np
import scipy.stats
import math
from time import time
from tabulate import tabulate

# Parameter Values
S0 = 110.  # initial index level
K = 100.  # strike price
T = 1.0  # time-to-maturity
r = 0.01  # riskless short rate
sigma = 0.2  # volatility

M = 12 ; # number of time steps
dt = T / M;
I = 100000  # number of simulations

# Valuation function
def AsianCallSimPrice(S0, K, T, r, sigma, M, I, CV = False):
    dt = T / M
    S = np.zeros((M + 1, I))
    z = np.random.standard_normal((M + 1, I)) # pseudorandom numbers
    Savg = np.zeros(I)
    S[0] = S0
    S = S0 * np.exp(np.cumsum((r - 0.5 * sigma ** 2) * dt
                            + sigma * np.sqrt(dt) * z, axis=0))
    Savg = np.average(S, axis = 0)
    if CV == False:
        price = np.exp(-r * T) * np.sum(np.maximum(Savg - K, 0)) / I
        error = math.sqrt(np.var(np.maximum(Savg - K, 0))) / math.sqrt(I)
        result = (price, error)
    else: 
        Tvector = np.arange(dt, T + dt, dt)
        T_avg = Tvector.mean()
        i_vector = np.arange(1, 2*M + 1, 2)
        sigma_avg = math.sqrt(sigma ** 2 / ( M ** 2 * T_avg) 
                            * np.dot(i_vector, Tvector[::-1]))                        
        delta = .5 * (sigma ** 2 - sigma_avg ** 2)
        d = (math.log(S0/K) + (r - delta + .5 * sigma_avg ** 2) * T_avg) / \
                    (sigma_avg * math.sqrt(T_avg))
        GeomAsianCall = np.exp(-delta * T_avg) * S0 * scipy.stats.norm.cdf(d) - \
                     np.exp(-r * T_avg) * K * \
                     scipy.stats.norm.cdf(d - sigma_avg * math.sqrt(T_avg))
        S_CV = scipy.stats.mstats.gmean(S, axis = 0)
        X = np.exp(-r * T) * np.maximum(S_CV - K, 0)
        Y = np.exp(-r * T) * np.maximum(Savg - K, 0)
        b = np.cov(X,Y)[0][1] / X.var()
        price = Y.mean() - b * (X.mean() - GeomAsianCall)
        error = math.sqrt(np.var(Y - b*X)) / math.sqrt(I)
        rho = np.corrcoef(X,Y)[0][1]
        result = (price, error, rho)
    return result


# Calculating execution time
t0 = time()
CallPrice = AsianCallSimPrice(S0, K, T, r, sigma, M, I)
CallPriceCV = AsianCallSimPrice(S0, K, T, r, sigma, M, I,True)
tnp1 = time() - t0

# Results table
N_sim = [1000, 10000, 100000, 1000000]
res = []
for i in range(0, len(N_sim)):
    newrow = [N_sim[i], 
            AsianCallSimPrice(S0, K, T, r, sigma, M, N_sim[i])[0],
            AsianCallSimPrice(S0, K, T, r, sigma, M, N_sim[i])[1],
            AsianCallSimPrice(S0, K, T, r, sigma, M, N_sim[i], True)[0], 
            AsianCallSimPrice(S0, K, T, r, sigma, M, N_sim[i], True)[1],
            AsianCallSimPrice(S0, K, T, r, sigma, M, N_sim[i], True)[2]]
    res.append(newrow)

table = tabulate(res, headers=["N", "Price", "Error", 
                             "PriceCV", "ErrorCV", "Corr"])
print(table)
# Write to file
f = open('bsm_mcs_asian_output.txt', 'w')
f.write(table)
f.close()