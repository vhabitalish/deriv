import math
import numpy as np
import matplotlib.pyplot as plt

def funa(T1, T2, alpha, N):
    return np.array([ T2 + (T1 -T2)*math.exp(-alpha*i) for i in range(0,N)])

def main():
    plt.plot(funa(0.5,0.1,0.5,20))
    plt.show()


if __name__ == "__main__":
    main()

    

