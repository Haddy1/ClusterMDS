#!/usr/bin/python
from __future__ import print_function

from sklearn.decomposition import PCA
import numpy as np
import scipy
from scipy.spatial.distance import squareform
from sklearn.metrics import euclidean_distances

import matplotlib.pyplot as plt

#import libSMACOF_theano
#Multidimensional Scaling using the "Scaling by MAjorizing a COmplicated Function" (SMACOF) alghorithm (Borg, I.; Groenen, P. (1997), Modern Multidimensional Scaling: theory and applications, New York: Springer-Verlag.)
class SMACOF():

    #setup constants
    def __init__(self, data, n_components=2):
        self.n_components = n_components
        self.delta = self.calcDist(data)
        self.size_inv = 1.0/data.shape[0]

    def getInitValues(self, data):
        n_init = 4

        #first guess: pca_transform
        pca = PCA(self.n_components)
        init_best = pca.fit(data).transform(data)
        dist_best = self.fDist(init_best)
        s_best = self.sigma(dist_best)


        #try random initial values
        for k in range(0,4):
            init = np.random.uniform(0.000001, 10, (data.shape[0], self.n_components))
            dist = self.fDist(init)
            s = self.sigma(dist)
            if s < s_best:
                init_best = init
                dist_best = dist
                s_best = s

        return init_best, dist_best, s_best



    def calcDist(self, X):
        XX = np.dot(np.sum(X**2, axis=1)[:, np.newaxis], np.ones((1,X.shape[0])))
        #XX = np.sum(X**2, axis=1)
        YY = XX.T
        dist = np.dot(X, X.T)
        dist *= -2
        dist += XX
        dist += YY
        np.maximum(dist, 0, out=dist)
        return np.sqrt(dist)


    #Cost function to be minimized
    def sigma(self, distX):
        s = np.sum( np.square(np.subtract(distX, self.delta)))
        return s


    def bCalc2(self, distZ):
        B = np.zeros(distZ.shape)

        i = np.nonzero(distZ)

        B[i] = -self.delta[i] / distZ[i]
        B = B - np.diag(np.diag(B))
        d = np.sum(B, axis=0)
        B = B - np.diag(d)

        return B

    def bCalc(self, B, distZ):

        #Ignore divide by zero erros, we'll fix them later
        with np.errstate(divide='ignore', invalid='ignore'):

            ratio = np.divide(self.delta, distZ)
            #search for invalid values and set them to zero
            ratio[ ~ np.isfinite(ratio)] = 0

            B = -ratio

        B[np.diag_indices(B.shape[0])] += ratio.sum(axis=1)
        return B

    # "Guttman Transformation" of X, as replacement of the derivative of sigma
    def guttmanTrans(self ,X, Z, distZ, B):
        X = self.size_inv * np.dot(self.bCalc(B, distZ), Z)
        return X

    def solve(self, data, initX = None, eps = 1e-6, maxiter = 10000):
        theano_smac = libSMACOF_theano.SMACOF(data)
        self.fDist = theano_smac.fdist

        #check if initial guess was provided
        if np.array_equal(initX, None):
            X, distX, s = self.getInitValues(data)
        else:
            X = initX
            distX = self.fDist(X)
            s = self.sigma(distX)

        dist_norm = np.sqrt((X**2).sum(axis=1)).sum()
        s = s / dist_norm



        Z = X
        distZ = distX
        B = np.zeros(distZ.shape)


        iter=1

        for k in range(1,maxiter):
            s_old = s
            X = self.guttmanTrans(X, Z, distZ, B)
            #distX = self.calcDist(distX, X)
            distX = self.fDist(X)
            dist_norm = np.sqrt((X**2).sum(axis=1)).sum()
            s = self.sigma(distX) / dist_norm
            if (s_old - s ) < eps:
                break
            Z = X
            distZ = distX

            iter+=1


        return (X)

#Principal component analysis
def pca(data, n_components = 2):
    pca = PCA(n_components= n_components)
    return (pca.fit(data).transform(data))
def mds(data):
    smacof= SMACOF(data, 2)
    return smacof.solve(data)

