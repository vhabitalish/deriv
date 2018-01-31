""" Module for Ornestein Uhlenback process with theta set to zero
"""

import math
import numpy as np
import numpy.random
import scipy.optimize
import matplotlib.pyplot as plt
import deriv.Correlatedpaths as corpath

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
        R = np.zeros([self.timeslices + 1, self.numpaths])
        t = self.time / self.timeslices
        R[0, :] = 0
        for i in range(1,self.timeslices + 1):
            for j in range(0, self.numpaths):
                alpha = self.meanrev[i-1]
                sigma = self.sigma[i-1]
                R[i, j] = (R[i-1, j]*math.exp(-alpha*t)
                    + paths[i-1,j] * sigma * math.sqrt((1 - 
                    math.exp(-2*alpha*t))/(2*alpha)))           
        self.paths = R + np.reshape(initts,(self.timeslices+1,1))
    
    def fwdsonpath(self,i,j,k):
        r = self.paths[i,j] *np.ones([self.N - i])
        t = self.T / self.N
        for p in range(i+1,self.N):
            alpha = self.meanrev[p-1]
            r[p-i] = initts[p-i] 
                    + (r[p-i-1] - initts[p-i-1])*math.exp(-alpha*t)                
        return r[:k] 

def swap(R,K):
    n = R.shape[0]
    df = 1
    pv = 0
    for i in range(0,n):
        df = df * 1/(1+R[i])
        pv = pv + (R[i]-K)*df
    return pv

def parswap(R):
    n = R.shape[0]
    df = 1
    pv = 0
    level = 0
    for i in range(0,n):
        df = df * 1/(1+R[i])
        pv = pv + (R[i])*df
        level = level + df
    return pv/level, level

   
def ex3():

    numpaths = 10000 
    N = 15
    np.random.seed(100910)
    R0term = np.concatenate((np.linspace(0.071,0.095,10), np.ones([N-10+1])*0.095))
    R00 = np.zeros(N+1)
    #R00 = R0term
    meanrev = np.concatenate((np.linspace( -0.5,-0.005,3), np.ones([N-3])*0.00005))
    #meanrev = np.ones([N]) *  0.05
      
    theta = np.ones([N]) * 0.05  

    sigma = np.concatenate(
        (np.linspace(0.007,0.015,3),np.linspace(0.015,0.012,3),np.ones([N-6])*0.012)
        ) 
    T = 20
       
    paths = corpath.getpaths(np.array([[1]]), numpaths, N)
    lgmm = OrUhl(R0term,meanrev, sigma, paths[0], T)
    R = lgmm.paths    
    numpaths=R.shape[1]

    #plot 10 random sample paths    
    a=np.random.permutation(R.shape[1])

    for i in range(0,30):
        plt.plot(R[:,a[i]])
    plt.show()

    return lgmm
 

if __name__ == "__main__":
    lgmm = ex3()
