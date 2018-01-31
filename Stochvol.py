""" Module for Stoch Vol process
"""

import math
import numpy as np
import numpy.random
import matplotlib.pyplot as plt
import deriv.Correlatedpaths as corpath
import deriv.Oruhl as irprocess
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
            # do drift adjustment
            driftadj = (np.mean(R[i])/np.mean(R[i-1])*
                    (1+t*np.mean(rfor[i-1]))/(1+t*np.mean(rdom[i-1])))
            R[i] = R[i]/driftadj        
        self.paths = R
    
    def optprice(self,strike, callput):
        payoff = np.mean(np.maximum(callput*(self.paths[-1,:] - strike),0))
        return payoff
   
def main():
    
    np.random.seed(100910)
    numpaths = 100000 
    N = 20
    rho = 0.75
    T = 10.0
    t = T/N

    # 0 :fx, 1:fxvol, 2:dom, 3:for 
    rho01 = -0.7
    rho02 = -0.7
    rho03 = 0
    rho12 = 0
    rho13 = 0
    rho23 = 0

    cov = np.array([[1, rho01, rho02, rho03],
                    [rho01, 1, rho12, rho13],
                    [rho02, rho12, 1, rho23],
                    [rho03, rho13, rho23, 1]])
                     
    
    paths = corpath.getpaths(cov, numpaths, N)
    
    #dom
    domts = np.ones(N+1) * 0.12 
    meanrev = np.ones([N])*0.1
    sigma = np.ones([N])*0.02
    rdom = irprocess.OrUhl(domts,meanrev, sigma, paths[2], T)    

    #for
    forts = np.ones(N+1) * 0.02 
    meanrev = np.ones([N])*0.1
    sigma = np.ones([N])*0.007
    rfor = irprocess.OrUhl(forts,meanrev, sigma, paths[3], T)    


    #Stoch vol process
    meanrev = np.ones([N])*0.1
    vvol = np.ones([N])*0.15
    basevol = np.ones([N+1])*0.1
    vol = volprocess.Logoruhl(basevol, meanrev, vvol, paths[1], T)
    spot = 3.75
    sigma = vol.paths
    asset = Stochvol(spot, sigma, paths[0], T, rdom.paths, rfor.paths)
    #plot 10 random sample paths    
    R = asset.paths
    
    a=np.random.permutation(R.shape[1])
    for i in range(0,50):
        plt.plot(rdom.paths[:,a[i]])
    plt.title("Dom Rate Process")
    plt.show()

    for i in range(0,50):
        plt.plot(rfor.paths[:,a[i]])
    plt.title("For Rate Process")
    plt.show()

    for i in range(0,50):
        plt.plot(vol.paths[:,a[i]])
    plt.title("Vol Process")
    plt.show()

    for i in range(0,50):
        plt.plot(R[:,a[i]])
    plt.title("Fx Process")
    plt.show()

    fxfwds = np.ones([N+1])*spot    
    fxfwdspaths = np.ones([N+1])*spot    
    fxvolpaths = np.ones([N+1])*basevol[0]    
    
    for i in range(1,N+1):
        fxfwds[i] = fxfwds[i-1]*(1+t*np.mean(rdom.paths[i-1]))/(1+t*np.mean(rfor.paths[i-1]))
        fxfwdspaths[i] = np.mean(R[i])
        fxvolpaths[i] = np.std(np.log(R[i]))/math.sqrt(t*i)
    
    plt.plot(fxfwds)
    plt.plot(fxfwdspaths)
    plt.title("Fxfwds: fwds and on paths")
    plt.show()

    plt.plot(fxvolpaths)
    plt.title("Vol evolving with time")    
    plt.show()

    fwd = fxfwds[-1]
    std = fxvolpaths[-1] * fwd
    x = np.linspace(fwd - 2 * std, fwd + 2 * std,20)
    y = [ BS.blackimply(fwd,stri,T,1,asset.optprice(stri,1)) for stri in x]
    plt.plot(x,y)
    plt.title("Terminal Smile")
    plt.show()
    print(BS.blackimply(fwd, fwd +std, T, 1,(asset.optprice(fwd+std,1))))
    print(BS.blackimply(fwd, fwd, T, 1,(asset.optprice(fwd,1))))
    print(BS.blackimply(fwd, fwd - std, T, -1,(asset.optprice(fwd-std,-1))))
    return asset, vol
    

if __name__ == "__main__":
    asset, vol = main()
