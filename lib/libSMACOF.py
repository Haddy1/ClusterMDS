#!/usr/bin/python
from __future__ import print_function

from sklearn.decomposition import PCA
import numpy as np
import scipy
from scipy.spatial.distance import squareform
from sklearn.metrics import euclidean_distances

import matplotlib.pyplot as plt

class SMACOF():
    """
    Multidimensional Scaling

    Multidimensional Scaling using the "Scaling by MAjorizing a COmplicated Function" (SMACOF) alghorithm

    Parameters
    ---------
    data: array-like
        array containing high dimensional data points

    n_components: int, optional, default: 2
        number of dimensions to which the data should be transformed

    maxiter: int, optional, default: 10000
        Maximum number of iterations of the SMACOF Alghorithm

    References
    ---------
    Borg, I.; Groenen, P. (1997), Modern Multidimensional Scaling: theory and applications, New York: Springer-Verlag.
        """


    #setup constants
    def __init__(self, data, n_components=2, maxiter = 10000):
        self.n_components = n_components
        self.maxiter = 10000
        self.delta = self.calcDist(data)
        self.size_inv = 1.0/data.shape[0]

    def getInitValues(self, data):
        """
        Provides initial values

        Parameters
        ----------
        data: array-like
            high dimensional data

        Returns
        -------
        init_best: array-like
            Guess for initial low dimensional data
        dist_best: array-like
            Initial Distance Matrix for init_best
        s_best: double
            Initial Result of Cost Function sigma
            """
        n_init = 4

        #first guess: pca_transform
        pca = PCA(self.n_components)
        init_best = pca.fit(data).transform(data)
        dist_best = self.calcDist(init_best)
        s_best = self.sigma(dist_best)


        #try random initial values
        for k in range(0,4):
            init = np.random.uniform(0.000001, 10, (data.shape[0], self.n_components))
            dist = self.calcDist(init)
            s = self.sigma(dist)
            if s < s_best:
                init_best = init
                dist_best = dist
                s_best = s

        return init_best, dist_best, s_best



    def calcDist(self, X):
        """
        Calculates Distance Matrix

        Parameters
        ---------
        X: array-like
            Input Array
        Returns
        --------
        dist: array-like
            squared symmetric array containing the euclidian distances between each row of X
            """


        XX = np.dot(np.sum(X**2, axis=1)[:, np.newaxis], np.ones((1,X.shape[0])))
        #XX = np.sum(X**2, axis=1)
        YY = XX.T
        dist = np.dot(X, X.T)
        dist *= -2
        dist += XX
        dist += YY
        np.maximum(dist, 0, out=dist)
        return np.sqrt(dist)


    def sigma(self, distX):
        """
        Cost Function to be minimized

        Parameters
        --------
        distX: array-like
            distance matrix of low dimensional X for current iteration step
        Returns
        ------
        s: float
            squared difference of high- and lowdimensional distances
            """


        s = np.sum( np.square(np.subtract(distX, self.delta)))
        return s



    def bCalc(self, B, distZ):
        """
        Calculates B
        """

        #Ignore divide by zero erros, we'll fix them later
        with np.errstate(divide='ignore', invalid='ignore'):

            ratio = np.divide(self.delta, distZ)
            #search for invalid values and set them to zero
            ratio[ ~ np.isfinite(ratio)] = 0

            B = -ratio

        B[np.diag_indices(B.shape[0])] += ratio.sum(axis=1)
        return B

    def guttmanTrans(self ,X, Z, distZ, B):
        """
        Guttman Transformation: Update Function for X
        """

        X = self.size_inv * np.dot(self.bCalc(B, distZ), Z)
        return X

    def solve(self, data, initX = None, eps = 1e-6):
        """
        Performs Multidimensional Scaling

        Parameters
        ---------
        data: array-like
            high dimensional data to be transformed in low dimensional form
        initX: array-like, optional
            Initial Guess for low dimensional data
            default is PCA transformed data or random data, depending which has better stress
        eps: float
            convergence tolerance, w.r.t sigma

        Returns
        -------
        X: array-like
            low dimensional data
            """

        #check if initial guess was provided
        if np.array_equal(initX, None):
            X, distX, s = self.getInitValues(data)
        else:
            X = initX
            distX = self.calcDist(X)
            s = self.sigma(distX)

        dist_norm = np.sqrt((X**2).sum(axis=1)).sum()
        s = s / dist_norm
        Z = X
        distZ = distX
        B = np.zeros(distZ.shape)

        for k in range(1,self.maxiter):
            s_old = s
            X = self.guttmanTrans(X, Z, distZ, B)
            #distX = self.calcDist(distX, X)
            distX = self.calcDist(X)
            dist_norm = np.sqrt((X**2).sum(axis=1)).sum()
            s = self.sigma(distX) / dist_norm
            if (s_old - s ) < eps:
                break
            Z = X
            distZ = distX

        return (X)

