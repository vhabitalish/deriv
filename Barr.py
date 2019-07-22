#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 23:07:04 2019

@author: vishal
"""
import numpy as np
import deriv.Correlatedpaths as corpath
import math
import numpy.random
import matplotlib.pyplot as plt


def nohitprob(sim, vol, time, barr, barrud):
    """ path by path prob if bar was hit
        sim: paths shape(timeslices+1,numpaths)
        volspaths : localvol along(timeslice-1,numpaths)
        time : time for each slice shape ( timesslices)
        barr: barrier
        barrud : up 0 or down 1
        estimate prob of cross as (1 -exp(2/vol**2*Tln(barr/So)*ln(S/barr)))
    """
    numpaths = sim.shape[1]
    timeslice = sim.shape[0] - 1
    dti = time[1:] - time[:-1]
    nohitprobb = np.ones(numpaths)
    for i in range(0, timeslice):
        ti = dti[i]
        prob = np.exp(2.0/vol[i]**2/ti*np.log(barr/sim[i]) *np.log(sim[i+1]/barr))
        if barrud == 0:
            nohitprobb[sim[i] > barr] = 0
            nohitprobb[sim[i+1] > barr] = 0
            nohitprobb = nohitprobb*(1-prob)
        else:
            nohitprobb[sim[i] < barr] = 0
            nohitprobb[sim[i+1] < barr] = 0
            nohitprobb = nohitprobb*(1-prob)
    return np.sum(nohitprobb)/numpaths 


def testbarr(timeslices, barr,vol):
    
    T=1
    s = timeslices
    t =T/s
    numpaths= 50000
    fwd= 1.13
    #vol = 0.075
    
    x = np.random.normal(0,1,size=numpaths*s)
    y = np.reshape(x,(s,numpaths))
    z = np.ones((s+1,numpaths))*fwd
    for i in range(0,s):
        z[i+1] = z[i]*np.exp(-0.5*vol**2*t +y[i]*vol*math.sqrt(t))
     
        
    nohitprobb = np.ones(numpaths)
    for i in range(0, s):
        probt= np.exp(2.0/t/vol**2*np.log(barr/z[i])*np.log(z[i+1]/barr))
        nohitprobb[z[i] < barr] = 0
        nohitprobb[z[i+1] < barr] = 0
        nohitprobb = nohitprobb*(1-probt)
    
    
    maxz= np.min(z,axis=0)
    modbarr=barr*math.exp(0.5826*vol*math.sqrt(t))
    prob=np.sum(np.ones(numpaths)[maxz > barr])/ numpaths
    probmodbarr=np.sum(np.ones(numpaths)[maxz > modbarr])/ numpaths
    print(barr,modbarr)
    probcorrected = np.sum(nohitprobb)/numpaths 
    return (prob,probmodbarr,probcorrected)

def main():
    fwd0 = 1.13
    vol0 = 0.075
    T = 1
    N = 10
    TI = np.linspace(0,T,N+1)
    vol = np.ones((N,50000))*vol0
    X = corpath.getpaths(np.array([[1,0],[0,1]]),50000,N+1)
    sim = np.ones((N+1,50000))*fwd0
    dti = TI[1:] - TI[:-1]
    
    for i in range(0,N):
        ti= dti[i]
        sim[i+1] = sim[i]*np.exp(-0.5*vol0**2*ti + vol0*math.sqrt(ti)*X[0,i])

    a=np.random.permutation(50000)
    for i in range(0,50):
        plt.plot(TI,sim[:,a[i]])
    plt.show()
    p1 = nohitprob(sim,vol,TI,1.4,0)
    p2 = nohitprob(sim,vol,TI,1.1,1)
    return (sim,vol,TI) 

if __name__ == "__main__":
    (sim,vol,TI) = main()

    #print(blackimply(100,105,1,1,3.99))
