#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys
sys.path.append("../")

import theano
import theano.tensor as T
import numpy as np
import libMDS as mds

import lib.libSMACOF_theano
import lib.libSMACOF

eps = 1e-5
data = np.loadtxt('Kontrolle2_Ch_rms_w150_s30_ztrans.txt', dtype=np.float32)
#data = np.array([[0,0,0], [0,0,1], [0,1,0], [0,1,1],[1,0,0],[1,0,1], [1,1,0], [1,1,1]]).astype('float32')

pca_data = mds.pca(data).astype(np.float32)

size_inv = 1.0 / pca_data.shape[0]

smacof = libSMACOF_theano.SMACOF(data)

smalib = libSMACOF.SMACOF(data)
delta = smalib.delta
distX = smalib.calcDist(pca_data)
s = smalib.sigma( distX)
X0, distX_s, s_s = smacof.getInitValues(data)
delta_s = smacof.fdist(data)



print(distX.shape, distX_s.shape)
err_sigma = np.sum((s_s - s)**2)
err_dist= np.sum((distX_s - distX)**2)
err_delta = np.sum((delta_s - delta )**2)
print ("init")
print ("err_delta: " , err_delta)
print ("err_distX: " , err_dist)
print ("err_sigma: " , err_sigma)
delta = smalib.calcDist(data).astype(np.float32)
distX = smalib.calcDist(pca_data).astype(np.float32)
#finit = theano.function([dataT, pca_dataT], [distX_T, delta_T, s_old], profile=False)


#Define Theano functions calls
def calcB(distX, delta):
    delta_T = T.fmatrix()
    distX_T = T.fmatrix()
    B_T = T.fmatrix()
    B_T = smacof.bCalc( distX_T, delta_T)
    fB = theano.function([ distX_T, delta_T], B_T, profile=False, allow_input_downcast=True)
    return fB(distX, delta)

def guttmanTrans(X, distX, delta, size_inv):
    delta_T = T.fmatrix()
    distX_T= T.fmatrix()
    X_T = T.fmatrix()
    X_n = T.fmatrix()
    size_inv_T = T.fscalar()
    X_n = smacof.guttmanTrans(X_T, distX_T, delta_T, size_inv_T)
    fgutmann = theano.function([X_T, distX_T, delta_T, size_inv_T], X_n, profile=False, allow_input_downcast=True)
    return fgutmann(X, distX, delta, size_inv)



print ('calcB')
B = calcB(distX, delta)
B_s = smalib.bCalc(np.zeros(distX.shape), distX)
err_B = np.sum((B_s - B)**2)
print (err_B)
#if print_values:
#    print (B)
#    print (B_s)
print ('guttmanTrans')
X = guttmanTrans(pca_data, distX, delta, size_inv)
X_s = smalib.guttmanTrans(np.zeros(pca_data.shape), pca_data, distX, B_s)
err_guttman = np.sum((X_s - X)**2)
print(err_guttman)
print ('calcDist')
distX = smacof.fdist(X)
distX_s = smalib.calcDist( X)
err_dist= np.sum((distX_s - distX)**2)
print(err_dist)
#print (distX)
#print (distX_s)
print ('sigma')
s = smacof.fsigma(distX, delta)
s_s = smalib.sigma(distX)
err_sigma = np.sum((s_s - s)**2)
print(err_sigma)
distZ_s = distX
Z_s = X_s
print("loop")
for k in range(1,1000):
    B_s = np.zeros(distZ_s.shape)

    B = calcB(distX, delta)
    B_s = smalib.bCalc( distX_s)
    B = B_s
    err_B = np.sum((B_s - B)**2)
    #print (k, err_B)

    s_old_s = s_s
    X_s = smalib.guttmanTrans(X_s, Z_s, distZ_s, B_s)
    X = guttmanTrans(X, distX, delta, size_inv)
    X_s = X
    distX_s = smalib.calcDist( X_s)
    distX = calcDist(X)
    s = sigma(distX, delta)
    s_s = smalib.sigma( distX_s)
    errX = np.sum((X_s - X)**2)
    err_dist= np.sum((distX_s - distX)**2)
    print (k, err_dist)

    if (s_old_s - s_s) < eps:
        break
    s_old = s
    Z_s = X_s
    distZ_s = distX_s
