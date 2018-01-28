""" Module for generating correlated path name
"""

import numpy as np
import numpy.random

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

def main():
    """ Simple test function """
    cor = np.array([[1, 0.5], [0.5, 1]])
    numpaths = 10000
    timeslice = 20
    X = getpaths(cor, numpaths, timeslice)
    print(X.shape, np.cov(X[0, 10, :], X[1, 10, :]))
    print(np.cov(X[0, 10, :], X[0, 11, :]))
    print(np.cov(X[0, 10, :], X[1, 11, :]))
    return X


if __name__ == "__main__":
    X = main()
