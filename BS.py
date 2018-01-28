#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 23:35:55 2018

@author: vishal
"""
from math import sqrt,exp,log
from scipy.stats import norm
import scipy.optimize as opt


def black(v,F,K,T,cp):
    """ callorput = 1 call/ -1 put 
    """
    N = norm.cdf
    n = norm.pdf
    d1 = (log(F/K)+(0.5*v*v)*T)/(v*sqrt(T))
    d2 = d1-v*sqrt(T)
    price = cp*(F*N(cp*d1)-K*N(cp*d2))
    vega = F*sqrt(T)*n(d1)
    return price

def blackvega(F,K,T,v,cp):
    """ callorput = 1 call/ -1 put 
    """
    N = norm.cdf
    n = norm.pdf
    d1 = (log(F/K)+(0.5*v*v)*T)/(v*sqrt(T))
    vega = F*sqrt(T)*n(d1)
    return vega

def optblack(F,K,T,cp,prem):
    def boundblack(vol):
        return black(vol,F,K,T,cp) - prem
    return boundblack

def optblackvega(F,K,T,cp,prem):
    def boundblackvega(vol):
        return blackvega(F,K,T,vol,cp)
    return boundblackvega


def blackimply(F,K,T,cp,prem):
    return opt.newton(optblack(F,K,T,cp,prem), x0=0.1, fprime=optblackvega(F,K,T,cp,prem))
    return opt.newton(optblack(F,K,T,cp,prem), x0=0.1)

if __name__ == "__main__":
    print(blackimply(100,105,1,1,3.99))
