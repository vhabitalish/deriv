""" Module for Lognormal mean reverting process
"""

import math
import numpy as np
import numpy.random
import scipy.optimize
import matplotlib.pyplot as plt
import deriv.Correlatedpaths as corpath


class Logoruhl:
    """ Mean reverting log normal process 
        initts  : initial TS
        meanrev : mean reversion shape (timeslice,)
        sigma   : vol shape (timeslice,)
        paths : Weiner process paths shape (timeslice,numpaths)
        time : total time covered by N time steps
    """
    def __init__(self, initts, meanrev, sigma, paths, time):
        self.meanrev = meanrev
        self.sigma = sigma
        self.numpaths = paths.shape[1]
        self.timeslices = paths.shape[0]
        self.time = time     
        R = np.zeros([self.timeslices + 1, self.numpaths])
        t = self.time / self.timeslices
        R[0, :] = math.log(initts[0])
        for i in range(1,self.timeslices + 1):
            alpha = self.meanrev[i-1]
            sigma = self.sigma[i-1]
            vol = sigma * math.sqrt((1 -math.exp(-2*alpha*t))/(2*alpha))
            R[i] = (np.log(initts[i])
                    + (R[i-1] - np.log(initts[i-1])) *math.exp(-alpha*t)
                    - 0.5 * vol * vol * math.sqrt(t)
                    + paths[i-1] * vol)        
            '''
            for j in range(0, self.numpaths):

                R[i, j] = (np.log(initts[i])
                    + (R[i-1, j] - math.log(initts[i-1])) *math.exp(-alpha*t)
                    - 0.5 * vol * vol * math.sqrt(t)
                    + paths[i-1,j] * vol)        
            '''
        self.paths = np.exp(R)


def ex3():

    numpaths = 10000 
    N = 15
    np.random.seed(100910)
    initts = np.ones([N+1])*3.75
    meanrev = np.ones([N])*0.0001
    sigma = np.ones([N])*0.1
    T = 20
    paths = corpath.getpaths(np.array([[1]]), numpaths, N)
    lgmm = Logoruhl(initts, meanrev, sigma, paths[0], T)
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
