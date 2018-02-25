""" Module for Stoch Vol process
"""

import math
import numpy as np
import numpy.random
import numpy.linalg
import matplotlib.pyplot as plt
import deriv.Correlatedpaths as corpath
import deriv.Oruhl as irprocess
import deriv.Logoruhl as volprocess
import deriv.BS as BS
import deriv.volutil as vu
import scipy.interpolate as intp
import deriv.util as du
import scipy.optimize as opt

 
class Stochvol:
    """ LOG normal process with stoch vol
        Add a constant process to add non zero fwds
        sigma   : vol shape (timeslice, numpaths)
        paths : Weiner process paths shape (timeslice,numpaths)
        time : total time covered by N time steps
        rdom : set of rates (df = 1/(1+rT) same dim as paths 
        rfor : set of for rates same dim as paths
    """
    def __init__(self, spot, sigma, paths, time, rdom = None, rfor = None, volcalib = None ):
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
        cumppt = np.array([0.1,0.25,0.5,0.75,0.9])

        for i in range(1,self.timeslices + 1):
            print("calib timeslice",i)
            dfr = (1+t*np.mean(rfor[i-1]))/(1+t*np.mean(rdom[i-1]))
            fwd = np.mean(R[i-1])/dfr
            volji = np.maximum(0,sigma[i-1])
            cdf1 = volcalib[i-1].cumdf
            calibki = cdf1.getstrike(cumppt)*dfr
            volcalibi = volcalib[i-1].getvolforstrike(calibki)
            def calibslice(lvol):
                lvolf = np.maximum(0,lvol)
                interped = intp.interp1d(calibki,lvolf, kind =1, fill_value= "extrapolate" )
                lvolj = interped(R[i-1])
                volj = volji * lvolj
                #print(lvol)
                R[i] = R[i-1] *dffor[i-1]/dfdom[i-1]*np.exp(-volj*volj*t/2 + paths[i-1]*volj*math.sqrt(t))
                driftadj = np.mean(R[i])/fwd
                R[i] = R[i]/driftadj
                volsim = [BS.volfromsim(R[i],fwd,strk,i*t) for strk in calibki]
                return np.linalg.norm(volsim - volcalibi)

                cdf0 = vu.cdf(BS.mccdf(R[i]))
                return  np.linalg.norm(cdf0.probinterpinv(cumppt) - cdf1.probinterpinv(cumppt))
            # do drift adjustment
            res = opt.minimize(calibslice,[1.0,1.0,1.0,1.0,1.0], method="Nelder-Mead",options={"maxiter":75})
            #print(res.message, res.x)
            calibslice(res.x)
            #driftadj = np.mean(R[i])/fwd
            #R[i] = R[i]/driftadj
            if volcalib is not None:
                x = np.linspace(0.05,0.95,20)
                cdf0 = vu.cdf(BS.mccdf(R[i]))
                p = cdf0.getcumprob(R[i])
                adjRi  = cdf1.getstrike(p)
                plt.plot(cdf0.getstrike(x),x, label = "original")
                plt.plot(cdf1.getstrike(x),x, label = "target")
                plt.title("generated cdf vs target")
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
  
                plt.show()
                R[i] = adjRi
            driftadj = np.mean(R[i])/fwd
            R[i] = R[i]/driftadj
    
        self.paths = R
    
    def optprice(self,strike, callput, node = -1):
        payoff = np.mean(np.maximum(callput*(self.paths[node,:] - strike),0))
        return payoff
   
    
def quad(R,f,vol,t):
    a = 0.1
    b = -0.5
    c = 1.0
    stdev = vol*math.sqrt(t)
    x = max(-10,min(10,np.log(R/f)/stdev))
    #print(x,du.quad(x,a,b,c))
    return du.quad(x,a,b,c)
    
    
    
def main():
    
    #np.random.seed(100910)
    numpaths = 20000 
    N = 10
    rho = 0.75
    T = 5.0
    t = T/N
    spot = 29.00
    
    # 0 :fx, 1:fxvol, 2:dom, 3:for 
    rho01 = 0
    rho02 = -0.35
    rho03 = -0.7
    rho12 = 0
    rho13 = 0
    rho23 = 0

    cov = np.array([[1, rho01, rho02, rho03],
                    [rho01, 1, rho12, rho13],
                    [rho02, rho12, 1, rho23],
                    [rho03, rho13, rho23, 1]])
                     
    
    paths = corpath.getpaths(cov, numpaths, N)
    
    #dom
    domts = np.ones(N+1) * 0.001
    meanrev = np.ones([N])*0.01
    sigma = np.ones([N])*0.005
    rdom = irprocess.OrUhl(domts,meanrev, sigma, paths[2], T)    

    #for
    forts = du.funa(0.135,0.115,1.0,N+1)
    meanrev = np.ones([N])*0.05
    sigma = np.ones([N])*0.015
    rfor = irprocess.OrUhl(forts,meanrev, sigma, paths[3], T)    


    atm = intp.interp1d([0.5,5],[0.135, 0.195])
    rr25 = intp.interp1d([0.5,5],[-0.035, -0.075])
    rr10 = intp.interp1d([0.5,5],[-0.08, -0.125])
    fly25 = intp.interp1d([0.5,5],[0.005, 0.0125])
    fly10 = intp.interp1d([0.5,5],[0.02, 0.06])
    fwd = spot
    volcalib = []    
    for i in range(0,N):
        ti = (i+1)*t
        fwd = fwd*(1+domts[i]*t)/(1+forts[i]*t)
        #print( fwd,atm(ti),rr25(ti),fly25(ti),rr10(ti),fly10(ti),ti)
        voluninterp = vu.fivepointsmile(fwd,atm(ti),rr25(ti),fly25(ti),rr10(ti),fly10(ti),ti)
        volinterp = vu.strikevolinterp(voluninterp)
        volcalib.append(vu.logvolslice(volinterp, fwd, ti))
        
    #Stoch vol process
    meanrev = np.ones([N])*0.1
    vvol = np.ones([N])*0.2
    basevol = np.ones([N+1])*0.135
    vol = volprocess.Logoruhl(basevol, meanrev, vvol, paths[1], T)
    sigma = vol.paths
    asset = Stochvol(spot, sigma, paths[0], T, rdom.paths, rfor.paths, volcalib )
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
        fxvolpaths[i] = BS.blackimply(fxfwdspaths[i],fxfwdspaths[i],t*i,1,asset.optprice(fxfwdspaths[i],1,i))
    
    plt.plot(fxfwds)
    plt.plot(fxfwdspaths)
    plt.title("Fxfwds: fwds and on paths")
    plt.show()

    plt.plot(fxvolpaths)
    plt.title("Vol evolving with time")    
    plt.show()

    fwd = fxfwds[-1]
    std = fxvolpaths[-1] * fwd
    x = np.linspace(fwd - 1.5 * std, fwd + 1.5 * std, 15)
    y = [ BS.blackimply(fwd,stri,T,-1,asset.optprice(stri,-1)) for stri in x]
    plt.plot(x,y)
    plt.title("Terminal Smile")
    plt.show()
    print(BS.blackimply(fwd, fwd +std, T, 1,(asset.optprice(fwd+std,1))))
    print(BS.blackimply(fwd, fwd, T, 1,(asset.optprice(fwd,1))))
    print(BS.blackimply(fwd, fwd - std, T, -1,(asset.optprice(fwd-std,-1))))
    return asset, vol, rdom, rfor, volcalib


def kiko(asset, vol, rdom, rfor):
    timesteps = asset.paths.shape[0] - 1
    numpaths = asset.paths.shape[1]
    
    # cpn  if fx > cpnk else 0
    cpn = 0.1
    cpnk = asset.paths[0,0] - 5
    
    # dom libor margin 
    lmar = 0.001
    
    #payatend
    redk = asset.paths[0,0] - 5
    
    
    
    
    

if __name__ == "__main__":
    asset, vol, dom, fgn, volcalib = main()
