""" Module for Ornestein Uhlenback process with theta set to zero
"""

import math
import numpy as np
import numpy.random
import scipy.optimize
import matplotlib.pyplot as plt
import deriv.Correlatedpaths as corpath
import deriv.util as du
import random

class OrUhl:
    """ Mean reverting normal process with constant  term structue
        initts : initial ts shape(timeslice +1,)
        meanrev : mean reversion shape (timeslice,)
        sigma   : vol shape (timeslice,)
        paths : Weiner process paths shape (timeslice,numpaths)
        time : total time covered by N time steps
    """
    def __init__(self,initts, meanrev, sigma, paths, time):
        self.meanrev = meanrev
        self.sigma = sigma
        self.numpaths = paths.shape[1]
        self.timeslices = paths.shape[0]
        self.time = time
        self.initts = initts  
        R = np.zeros([self.timeslices + 1, self.numpaths])
        t = self.time / self.timeslices
        R[0, :] = 0
        for i in range(1,self.timeslices + 1):
            alpha = self.meanrev[i-1]
            sigma = self.sigma[i-1]
            R[i] = (R[i-1]*math.exp(-alpha*t)
                    + paths[i-1] * sigma * math.sqrt((1 - 
                    math.exp(-2*alpha*t))/(2*alpha)))           
        '''       
            for j in range(0, self.numpaths):
                R[i, j] = (R[i-1, j]*math.exp(-alpha*t)
                    + paths[i-1,j] * sigma * math.sqrt((1 - 
                    math.exp(-2*alpha*t))/(2*alpha)))           
        '''
        self.paths = R + np.reshape(initts,(self.timeslices+1,1))
    
    def fwdsonpath(self,i,j,k):
        r = self.paths[i,j] *np.ones([self.timeslices - i])
        t = self.time / self.timeslices
        for p in range(i+1,self.timeslices):
            alpha = self.meanrev[p-1]
            r[p-i] = (self.initts[p-i] 
                    + (r[p-i-1] - self.initts[p-i-1])*math.exp(-alpha*t))
        return r[:k] 

def swap(R,K,t=1):
    n = R.shape[0]
    df = 1
    pv = 0
    for i in range(0,n):
        df = df * 1/(1+R[i]*t)
        pv = pv + (R[i]-K)*t*df
    return pv

def parswap(R, t=1):
    n = R.shape[0]
    df = 1
    pv = 0
    level = 0
    for i in range(0,n):
        df = df * 1/(1+R[i]*t)
        pv = pv + (R[i]*t)*df
        level = level + df
    return pv/level/t, level

   
def ex3():

    numpaths = 20000 
    N = 40
    T = 20.0
    t= T/N
    np.random.seed(100910)
    
  
    #R0term = np.concatenate((np.linspace(0.071,0.095,10), np.ones([N-10+1])*0.095))
    #R00 = np.zeros(N+1)
    #R00 = R0term
    #meanrev = np.concatenate((np.linspace( -0.5,-0.005,3), np.ones([N-3])*0.00005))
    #meanrev = np.ones([N]) *  0.05
   
    paths = corpath.getpaths(np.eye(2), numpaths, N)
    initts0 = np.ones(N+1) * 0.12 
    meanrev0 = du.funa(-2.5, 0, 0.5, N)
    sigma0 = du.funa(0.0007, 0, 2, N)
    lgmm0 = OrUhl(initts0, meanrev0, sigma0, paths[0], T)
    
    initts1 = np.ones(N+1) * 0.0001
    meanrev1 = du.funa(0.01, 0.01, 0.5, N)
    sigma1 = du.funa(0.005, 0.01, 1.0, N)
    lgmm1 = OrUhl(initts1, meanrev1, sigma1, paths[0], T)
    
    
  
    #plot 10 random sample paths    
    a=np.random.permutation(lgmm0.paths.shape[1])

    for i in range(0,30):
        plt.plot(lgmm0.paths[:,a[i]]+ lgmm1.paths[:,a[i]])
    plt.title("evolution of r")
    plt.show()

    
    for expiry in range(1,min(N - 10, 20),2):
        print("{:03}".format(expiry), end=":")
        for term in range(1, min(N - expiry,20),2):
            swappath = np.ones([numpaths])
            for j in range(0, numpaths):    
                swappath[j] = parswap(
                        lgmm0.fwdsonpath(expiry,j,term)
                        + lgmm1.fwdsonpath(expiry,j,term),t)[0]
            print("{:03.2f}".format(np.std(swappath)/math.sqrt(expiry*t)*100), end="  ")
        print()
    
 

if __name__ == "__main__":
    lgmm = ex3()
