#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 21:43:49 2018

@author: vishal
"""
import deriv.BS as BS
import deriv.util as du
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import math
import scipy.optimize as opt
import scipy.stats as stats


class logvolslice:
    def __init__(self, strikevol, fwd, mat):
        """ strike vol : 2d np array ascending order strike, vol
        """
        self.strikevol = strikevol
        self.fwd = fwd
        self.mat = mat
        self.strikevolinter = interpolate.interp1d(strikevol[:,0], strikevol[:,1], kind = 1, fill_value = "extrapolate")
        nofstrikes = self.strikevol.shape[0]
        strprob = np.ones((nofstrikes-1,2))*0.5
        for i in range(0,nofstrikes-1):
            v0 = self.strikevol[i][1]
            v1 = self.strikevol[i+1][1]
            k0 = self.strikevol[i][0]
            k1 = self.strikevol[i+1][0]
            #cp = -1 if k0 < self.fwd else 1
            cp = 1 if fwd < k0 else -1
   
            d0 = BS.black(v0,self.fwd,k0,self.mat,cp)
            d1 = BS.black(v1,self.fwd,k1,self.mat,cp)
            strprob[i,0] = (k1+k0)/2
            strprob[i,1] = (cp+1)/2+(d1-d0)/(k1-k0)
        self.cumdf = cdf(strprob)
        
    def getvolforstrike(self, strike):
        return self.strikevolinter(strike)
            
    def deltastrike(self, delta, cp):
        def bounddelta(strike):
            f = self.fwd
            v = self.getvolforstrike(strike)
            mat = self.mat
            err = (BS.blackdelta(v,f,strike,mat,cp) - delta)
            return err *err
        res = opt.minimize(bounddelta,self.fwd)
        return res.x
    def getrr(self,delta):
        c = self.getvolforstrike((self.deltastrike(delta,1)))
        p = self.getvolforstrike((self.deltastrike(delta,-1)))
        print(c,p)
        return c -p
    def getfly(self,delta):
        c = self.getvolforstrike((self.deltastrike(delta,1)))
        p = self.getvolforstrike((self.deltastrike(delta,-1)))
        fv = self.getvolforstrike(self.fwd)
        return c/2+p/2-fv
  
    
    
class cdf:
    def __init__(self,strikeprob):
        self.strikeprob = strikeprob
        self.probinterp = interpolate.interp1d(np.log(strikeprob[:,0]),strikeprob[:,1], kind = 3, fill_value = "extrapolate")
        self.probinterpinv = interpolate.interp1d(strikeprob[:,1],np.log(strikeprob[:,0]), kind = 3, fill_value = "extrapolate")
    def getcumprob(self,strike):
        '''
        if strike > self.strikeprob[-1,0]:
            return self.strikeprob[-1,1]
        if strike < self.strikeprob[0,0]:
            return self.strikeprob[0,1]
        '''    
        return self.probinterp(np.log(strike))
    def getstrike(self,prob):
        return np.exp(self.probinterpinv(prob))
    def getpdf(self):
        strikeprob = self.strikeprob
        strikeden = np.zeros((strikeprob.shape[0] - 1, 2))
        for i in range(0,self.strikeprob.shape[0] - 1) :
            strikeden[i,0] = 0.5*(strikeprob[i,0] + strikeprob[i+1, 0])
            strikeden[i,1] = ((strikeprob[i+1,1] - strikeprob[i,1])
                / (strikeprob[i+1,0] - strikeprob[i,0]))
        return strikeden
            
    def getsimulation(self,size):
        ''' use MC '''
        #x = np.random.uniform(size = size)
        x = np.random.uniform(size = int(size/2))
        x = np.concatenate((x, 1-x))
        '''
        
        x = np.random.normal(size = size)
        vols = 0.2
        fwd = 15.0
        T = 10
        sim = fwd*np.exp(-vols*vols*T/2+vols*x*math.sqrt(T))

        cdf1 = cdf(BS.mccdf(x))
        '''
        sim = self.getstrike(x)
        return sim    
   
    def getstrikevol(self,strikes, mat):
        simul = self.getsimulation(500000)
        fwd = simul.mean()
        print("Fwd",fwd)
        fwd = 15.0
        strikevol = np.zeros((strikes.shape[0],2))
        strikevol[:,0] = strikes
        strikevol[:,1] = np.array([ BS.volfromsim(simul,fwd,stri,mat) for stri in strikes] )
        return strikevol
    
def tracecallback(xk):
    print(".",xk)
    
def fivepointsmile(fwd, atm, rr25, fly25, rr10, fly10, mat):
    c25 = 0.5*(rr25 + 2*fly25 + 2*atm)
    p25 = c25 - rr25
    c10 = 0.5*(rr10 + 2*fly10 + 2*atm)
    p10 = c10 - rr10
    kc25 = BS.blackstrikefordelta(c25, fwd, mat, 1, 0.25)
    kc10 = BS.blackstrikefordelta(c10, fwd, mat, 1, 0.1)
    kp25 = BS.blackstrikefordelta(p25, fwd, mat, -1, 0.25)
    kp10 = BS.blackstrikefordelta(p10, fwd, mat, -1, 0.1)
 
    strikevol = np.ones((5,2))
    strikevol[0,0] = kp10
    strikevol[0,1] = p10
    strikevol[1,0] = kp25
    strikevol[1,1] = p25
    strikevol[2,0] = fwd
    strikevol[2,1] = atm
    strikevol[3,0] = kc25
    strikevol[3,1] = c25
    strikevol[4,0] = kc10
    strikevol[4,1] = c10
    
    return strikevol

def strikevolinterp(strikevol):
    points = 1000
    strikes = np.linspace(strikevol[0,0],strikevol[-1,0],points)
    interfn = interpolate.interp1d(strikevol[:,0], strikevol[:,1], fill_value = "extrapolate",kind = 2)
    interpstrikesmile = np.zeros([points,2])
    for i in range(0,points):
        interpstrikesmile[i,0] = strikes[i]
        interpstrikesmile[i,1] = interfn(strikes[i])
    return interpstrikesmile
 
def fivepointtostrikevol(fwd, atm, rr25, fly25, rr10, fly10, mat):
    ''' create volslice with atm vol
    transform strikes with a quadratic
    '''    
    strikevol = fivepointsmile(fwd, atm, rr25, fly25, rr10, fly10, mat)
    strikevolbase = fivepointsmile(fwd, atm, 0, 0, 0, 0, mat)
    
    strikevolmap  = interpolate.interp1d(np.log(strikevolbase[:,0]),np.log(strikevol[:,0]), fill_value = "extrapolate",kind = 2 )
    
    lbound = BS.blackstrikefordelta(atm, strikevol[0,0], mat, -1, 0.01)
    ubound = BS.blackstrikefordelta(atm, strikevol[-1,0], mat,  1, 0.01)
    stvol = np.ones([1000,2])
    stvol[:,0] = np.linspace(lbound , ubound, 1000)
    stvol[:,1] = atm
    vol = logvolslice(stvol, fwd, mat)
    baseprob = vol.cumdf.strikeprob
    prob = baseprob.copy()
    prob[:,0] = [np.exp(strikevolmap(np.log(stri))) for stri in baseprob[:,0]]
    prob[:,1] = baseprob[:,1]
    plt.plot(prob[:,0],prob[:,1],label ="smile")
    plt.plot(baseprob[:,0],baseprob[:,1],label ="nonsmile")
    plt.title("smile")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.show()
    
    strikevol = cdf(prob).getstrikevol(prob[:,0], mat)
    return strikevol

def testquadsmile():
    fwd = 15.0
    T = 10
    a =0.5
    b= 0.0001
    c =1.05
    stvol = np.ones([1000,2])
    stvol[:,0] = np.exp(np.linspace(np.log(0.5),np.log(100),1000))
    stvol[:,1] = 0.2
    stvol[:,1] = 0.2*du.quad(np.log(stvol[:,0]/fwd)/0.2/T,a,b,c)
   
    
    vol = logvolslice(stvol,fwd,T)
    cdfvol = vol.cumdf.strikeprob
    plt.plot(cdfvol[:,0],cdfvol[:,1])
    plt.title("cdf")
    plt.show()
    
    
    pdfvol = vol.cumdf.getpdf()
    plt.plot(pdfvol[:,0],pdfvol[:,1])
    plt.title("pdf")
    plt.show()
    strvol = vol.cumdf.getstrikevol(stvol[:,0], vol.mat)
    plt.plot(stvol[:,0], stvol[:,1], label = "input quadvoL")
    plt.plot(strvol[:,0], strvol[:,1], label ="vol from cdf")
    plt.title("vol from cdf")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
    
    
    xran = np.random.normal(size=10000)
    vols = 0.2
    sim = fwd*np.exp(-vols*vols*T/2+vols*xran*math.sqrt(T))
    
    lbound = vol.cumdf.getstrike(0.01)
    ubound = vol.cumdf.getstrike(0.99)
    print(lbound,ubound)
    x = np.linspace(ubound , lbound, 30)
    y = [ BS.volfromsim(sim, sim.mean(), stri, T) for stri in x]
    plt.plot(x,y)
    plt.title("vol from cdf")
    plt.show()
    
    print(vol.deltastrike(0.25,1), vol.deltastrike(0.25,-1))
    print(vol.getrr(0.25), vol.getfly(0.25))
    print(vol.deltastrike(0.1,1), vol.deltastrike(0.1,-1))
    
    print(vol.getrr(0.1), vol.getfly(0.1))
    
    strikevolsmile = fivepointtostrikevol(15,0.20,-0.06,0.01,-0.12,0.03,5)
    plt.plot(strikevolsmile[:,0], strikevolsmile[:,1])
    plt.title("interpolated smile")
    plt.show()
    
    return vol

def main():
    fwd = 15
    T = 5
    voluninterp = fivepointsmile(fwd,0.21,-0.079,0.0154,-0.1628,0.0515,T)
    volinterp = strikevolinterp(voluninterp)
    #volinterp = fivepointtostrikevol(fwd,0.20,-0.06,0.01,-0.11,0.05,T)
    plt.plot(voluninterp[:,0], voluninterp[:,1])
    plt.title("un interpolated smile")
    plt.show()
    
    
    
    plt.plot(volinterp[:,0], volinterp[:,1])
    plt.title("interpolated smile")
    plt.show()
    
    
    vol1 = logvolslice(volinterp, fwd, T)
    print(vol1.deltastrike(0.25,1), vol1.deltastrike(0.25,-1))
    print(vol1.getrr(0.25), vol1.getfly(0.25))
    print(vol1.deltastrike(0.1,1), vol1.deltastrike(0.1,-1))
    print(vol1.getrr(0.1), vol1.getfly(0.1))
    x = np.linspace(0.05,0.95,10)
    strikes  = vol1.cumdf.getstrike(x)
    plt.plot(strikes,vol1.getvolforstrike(strikes), label = "from slice")
    plt.plot(strikes,vol1.cumdf.getstrikevol(strikes,T)[:,1], label = "from dist")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.show()
    
    
    print(vol1.cumdf.strikeprob)
    
    return vol1

if __name__ == "__main__":
     vs = main()
