#!/usr/bin/python
from __future__ import print_function

import numpy as np
from sklearn.decomposition import PCA

import theano
import theano.tensor as T

#Multidimensional Scaling using the "Scaling by MAjorizing a COmplicated Function" (SMACOF) alghorithm (Borg, I.; Groenen, P. (1997), Modern Multidimensional Scaling: theory and applications, New York: Springer-Verlag.)
class SMACOF():
    maxiter = 10000
    mode = 'all'
    fstep = None
    floop = None

    def __init__(self, data, n_components = 2, maxiter = 10000, mode='all', floatX='float32'):
        theano.config.floatX = floatX
        self.n_components = n_components
        self.maxiter = 10000
        self.mode = mode
        self.initTheano()
        self.delta = self.fdist(data)
        self.size_inv = 1.0 / data.shape[0]

    def initTheano(self):
        X = T.matrix()
        dist = T.matrix()
        dist = self.calcDist(X)
        self.fdist = theano.function(inputs=[X], outputs = dist, allow_input_downcast=True)

        distX = T.matrix()
        delta = T.matrix()
        s = T.scalar()
        s = self.sigma(distX, delta)
        self.fsigma = theano.function(inputs=[distX, delta], outputs = s, allow_input_downcast=True)

        if self.mode == 'all':
            self.init_loop()
        else:
            self.init_step()

    def init_step(self):
        Z = T.matrix()
        distZ = T.matrix()
        delta = T.matrix()
        size_inv = T.scalar()
        X = T.matrix()
        distX = T.matrix()
        s = T.scalar()
        s_old = T.scalar()
        X, distX, s = self.step(Z, distZ, s_old, delta, size_inv)

        self.fstep = theano.function(inputs=[Z, distZ, delta, size_inv], outputs=[X, distX, s], allow_input_downcast=True)
    def step(self, Z, distZ, s_old, delta, size_inv):

        #update X
        X = self.guttmanTrans(Z, distZ, delta, size_inv)
        distX = self.calcDist(X)
        dist_norm = T.sqrt((X**2).sum(axis=1)).sum()
        s = self.sigma(distX, delta) / dist_norm

        return X, distX, s

    def init_loop(self):
        Z = T.matrix()
        distZ = T.matrix()
        delta = T.matrix()
        size_inv = T.scalar()
        X = T.matrix()
        distX = T.matrix()
        s = T.scalar()
        s_old = T.scalar()
        eps = T.scalar()

        ([X, distX, s]), updates = theano.scan(fn = self.loop_step, outputs_info= [Z, distZ, s_old], non_sequences = [delta, size_inv, eps], n_steps=self.maxiter)
        self.floop= theano.function(inputs=[Z, distZ, s_old, delta, size_inv, eps], outputs=[X[-1]], allow_input_downcast=True)

    def loop_step(self, Z, distZ, s_old, delta, size_inv, eps):

            #update X
            X = self.guttmanTrans(Z, distZ, delta, size_inv)
            distX = self.calcDist(X)
            dist_norm = T.sqrt((X**2).sum(axis=1)).sum()
            s = self.sigma(distX, delta) / dist_norm
            s_diff = s_old - s
            condition = T.lt(s_diff, eps)
            until = theano.scan_module.until(condition)

            return [X, distX, s], until

    def getInitValues(self, data):
        n_init = 4

        #first guess: pca_transform
        pca = PCA(self.n_components)
        init_best = pca.fit(data).transform(data)
        dist_best = self.fdist(init_best)
        s_best = self.fsigma(dist_best, self.delta)


        #try random initial values
        for k in range(0,4):
            init = np.random.uniform(0.000001, 10, (data.shape[0], self.n_components))
            dist = self.fdist(init)
            s = self.fsigma(dist, self.delta)
            if s < s_best:
                init_best = init
                dist_best = dist
                s_best = s

        return init_best, dist_best, s_best

    #Cost function to be minimized
    def sigma(self, distX, delta):
        s = T.sum(T.square(distX - delta))
        return s


    #calculate the distance matrix
    def calcDist(self, X):
        XX = T.dot(T.sum(X**2, axis=1).dimshuffle(0,'x') , T.ones((1, T.shape(X)[0])))
        YY = XX.T
        dist = T.dot(X, X.T)
        dist *= -2
        dist += XX
        dist += YY
        dist = T.maximum(dist, 0)
        dist = T.sqrt(dist)
        return dist


    def bCalc(self, distZ, delta):
        B = -T.switch(T.eq(distZ, 0), 0, delta / distZ)

        #set diagonal to zero so we can easily sum up the Columns
        B = T.extra_ops.fill_diagonal(B, 0)
        ##calculate diagonal
        bColumnSum = T.sum(B, axis = 1)
        B -= T.diag(bColumnSum)

        return B

    # "Guttman Transformation" of X, as replacement of the derivative of sigma
    def guttmanTrans(self, Z, distX,  delta, size_inv):
        X = size_inv* T.dot(self.bCalc(distX, delta), Z)
        return X



    def solve(self, data, initX = None, eps = 1e-6):

        #check if initial guess was provided
        if np.array_equal(initX, None):
            X, distX, s = self.getInitValues(data)
        else:
            X = initX
            distX = self.fdist(X)
            s = self.fsigma(distX, self.delta)

        dist_norm = np.sqrt((X**2).sum(axis=1)).sum()
        s = s / dist_norm

        Z = X
        distZ = distX

        #theano handling iteration
        if self.mode == 'all':
            X = self.floop(Z, distZ, s, self.delta, self.size_inv, eps)[-1]

        else:
            #theano function executed each iteration step
            for k in range(1,self.maxiter):
                s_old = s

                #call theano function
                X, distX, s = self.fstep(Z, distZ, self.delta, self.size_inv)
                if (s_old - s) < eps:
                    break
                Z = X
                distZ = distX

        return X
