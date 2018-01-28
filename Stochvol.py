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
    """
    def __init__(self, spot, sigma, paths, time):
        self.sigma = sigma
        self.numpaths = paths.shape[1]
        self.timeslices = paths.shape[0]
        self.sigma = sigma
        self.time = time        
        R = np.zeros([self.timeslices + 1, self.numpaths])
        t = self.time / self.timeslices
        R[0, :] = spot
        for i in range(1,self.timeslices + 1):
            for j in range(0, self.numpaths):
                vol = np.maximum(0,sigma[i-1, j] )
                R[i, j] = (R[i-1, j]*math.exp(-0.5*vol*vol*t
                    + paths[i-1,j] * vol * math.sqrt(t)))         
        self.paths = R
    
    def optprice(self,strike, callput):
        payoff = np.mean(np.maximum(callput*(self.paths[-1,:] - strike),0))
        return payoff
   
def main():
    
    np.random.seed(100910)
    numpaths = 300000 
    N = 15
    rho = 0
    T = 1
    paths = corpath.getpaths(np.array([[1, rho], [rho, 1]]), numpaths, N)
    meanrev = np.ones([N])*0.01
    vvol = np.ones([N])*0.2
    #Stoch vol process
    basevol = np.ones([N+1])*0.1
    vol = volprocess.Logoruhl(basevol, meanrev, vvol, paths[0], T)
    spot = 3.75
    sigma = vol.paths
    asset = Stochvol(spot, sigma, paths[1], T)
    #plot 10 random sample paths    
    R = asset.paths
    
    a=np.random.permutation(R.shape[1])
    for i in range(0,50):
        plt.plot(vol.paths[:,a[i]])
    plt.show()

    for i in range(0,50):
        plt.plot(R[:,a[i]])
    plt.show()
    x = np.linspace(2.75,5.5,20)
    y = [ BS.blackimply(spot,stri,T,1,asset.optprice(stri,1)) for stri in x]
    plt.plot(x,y)
    plt.show()
    print(BS.blackimply(spot,3.75, T, 1,(asset.optprice(3.75,1))))
    print(BS.blackimply(spot,4.5, T, 1,(asset.optprice(4.5,1))))
    print(BS.blackimply(spot,3.25, T, -1,(asset.optprice(3.25,-1))))
    return asset, vol
    

if __name__ == "__main__":
    asset, vol = main()
