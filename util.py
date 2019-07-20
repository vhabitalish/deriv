import math
import numpy as np
import matplotlib.pyplot as plt

def funa(T1, T2, alpha, N):
    return np.array([ T2 + (T1 -T2)*math.exp(-alpha*i) for i in range(0,N)])

def quad(x,a,b,c):
    return a*x*x+b*x+c

def main():
    plt.plot(funa(20.5,14.5,0.25,20))
    plt.show()
    x = np.linspace(-4,4,40)
    y = [ quad(xi,0.1,-.5,1) for xi in x]
    plt.plot(x,y)
    plt.show()
if __name__ == "__main__":
    main()

    

