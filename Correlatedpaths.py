""" Module for generating correlated path name
"""

import numpy as np
import numpy.random
import math

def getpaths(cor, numpaths, timeslices):
    """ Get correlated paths
        cor : correlation matrix
        numpaths : num of paths
        N : number of time slice
        returns array indexed asset,time slice, path)
    """
    noofassets = cor.shape[0]
    paths = np.random.multivariate_normal(
        np.zeros(noofassets), cor, size=int(numpaths*timeslices/2)).T
    paths = paths.reshape((noofassets, timeslices, -1))
    allpaths = np.concatenate((paths, -paths), axis=2)
    return allpaths

def threeasset(fwd0,vol0,fwd1,vol1,fwd2,vol2,T,rho01,rho02,rho12):
    cov = np.array([[1, rho01, rho02],
                    [rho01, 1, rho12],
                    [rho02, rho12, 1]])
    numpaths=10000
    timeslice=1    
    X = getpaths(cov, numpaths, timeslice)
    asset=np.zeros([3,numpaths])
    asset[0] = fwd0 *np.exp(-0.5*vol0**2*T + vol0*math.sqrt(T)*X[0])
    asset[1] = fwd1 *np.exp(-0.5*vol1**2*T + vol1*math.sqrt(T)*X[1])
    asset[2] = fwd2 *np.exp(-0.5*vol2**2*T + vol2*math.sqrt(T)*X[2])
    return asset
    
    """ Simple test function """
    cor = np.array([[1, 0.5], [0.5, 1]])
        p0 = np.maximum(sim[0]/k0-1,0)
    else :
        p0 = np.maximum(0,1-sim[0]/k0)
    if c1 :
def wof(sim,k0,k1,k2,c0,c1,c2):
    nopaths=sim.shape[1]
    if c0 :
        p2 = np.maximum(0,sim[2]/k2-1)
    print(np.sum(p0)/nopaths,np.sum(p1)/nopaths,np.sum(p2)/nopaths)
    wof=np.minimum(np.minimum(p0,p1),p2)
    wofp = np.sum(wof)/nopaths
    print(wofp)
    return wofp


def main():
        p1 = np.maximum(sim[1]/k1-1,0)
    else :
        p1 = np.maximum(0,1-sim[1]/k1)
    if c2 :
        p2 = np.maximum(sim[2]/k2-1,0)
    else :
    numpaths = 10000
    timeslice = 20
    X = getpaths(cor, numpaths, timeslice)
    print(X.shape, np.cov(X[0, 10, :], X[1, 10, :]))
    print(np.cov(X[0, 10, :], X[0, 11, :]))
    print(np.cov(X[0, 10, :], X[1, 11, :]))
    sim=threeasset(1/107.5/0.98,0.068,0.7067*1.005,0.075,1/6.876/1.01,0.045,1,0.3,0.3,0.5)
    wof(sim,1/104,0.7275,1/6.85,1,1,1)
    return sim


if __name__ == "__main__":
    X = main()
