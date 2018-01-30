""" Module for Stoch Vol process
"""

import math
import numpy as np
import numpy.random
import matplotlib.pyplot as plt
import deriv.Correlatedpaths as corpath
import deriv.Oruhl as volprocessN
import deriv.Logoruhl as volprocess

import BS
 
class Stochvol:
    """ LOG normal process with stoch vol
        Add a constant process to add non zero fwds
        sigma   : vol shape (timeslice, numpaths)
        paths : Weiner process paths shape (timeslice,numpaths)
        time : total time covered by N time steps
        rdom : set of rates (df = 1/(1+rT) same dim as paths 
        rfor : set of for rates same dim as paths
    """
    def __init__(self, spot, sigma, paths, time, rdom = None, rfor = None):
        self.sigma = sigma
        self.numpaths = paths.shape[1]
        self.timeslices = paths.shape[0]
        self.sigma = sigma
        self.time = time        
        R = np.zeros([self.timeslices + 1, self.numpaths])
        t = self.time / self.timeslices
        if rdom is None:
            rdom = np.ones((self.timeslices, self.numpaths)) * 0.05 
            rfor = np.ones((self.timeslices, self.numpaths)) * 0.05 
        
        dffor = 1.0/(1+rfor*t)
        dfdom = 1.0/(1+rdom*t)
        R[0, :] = spot
        for i in range(1,self.timeslices + 1):
            for j in range(0, self.numpaths):
                vol = np.maximum(0,sigma[i-1, j] )
                R[i, j] = (R[i-1, j]*dffor[i-1,j]/dfdom[i-1,j]
                    * math.exp(-0.5*vol*vol*t + paths[i-1,j] * vol * math.sqrt(t)))         
        self.paths = R
    
    def optprice(self,strike, callput):
        payoff = np.mean(np.maximum(callput*(self.paths[-1,:] - strike),0))
        return payoff
   
def main():
    
    np.random.seed(100910)
    numpaths = 100000 
    N = 10
    rho = 0
    T = 5
    paths = corpath.getpaths(np.array([[1, rho], [rho, 1]]), numpaths, N)
    meanrev = np.ones([N])*0.01
    vvol = np.ones([N])*0.2
    #Stoch vol process
    basevol = np.ones([N+1])*0.1
    vol = volprocess.Logoruhl(basevol, meanrev, vvol, paths[0], T)
    spot = 3.75
    sigma = vol.paths
    rdom = np.ones((N, numpaths)) * 0.12 
    rfor = np.ones((N, numpaths)) * 0.02 
    fwd = spot * math.pow(1+0.12*T/N,N)/math.pow(1+0.02*T/N,N)
    asset = Stochvol(spot, sigma, paths[1], T, rdom, rfor) 
    #plot 10 random sample paths    
    R = asset.paths
    
    a=np.random.permutation(R.shape[1])
    for i in range(0,50):
        plt.plot(vol.paths[:,a[i]])
    plt.show()

    for i in range(0,50):
        plt.plot(R[:,a[i]])
    plt.show()
    std = fwd*basevol[0]*math.sqrt(T)
    x = np.linspace(fwd - 2 * std, fwd + 2 * std,20)
    y = [ BS.blackimply(fwd,stri,T,1,asset.optprice(stri,1)) for stri in x]
    plt.plot(x,y)
    plt.show()
    print(BS.blackimply(fwd, fwd +std, T, 1,(asset.optprice(fwd+std,1))))
    print(BS.blackimply(fwd, fwd, T, 1,(asset.optprice(fwd,1))))
    print(BS.blackimply(fwd, fwd - std, T, -1,(asset.optprice(fwd-std,-1))))
    return asset, vol
    

if __name__ == "__main__":
    asset, vol = main()
