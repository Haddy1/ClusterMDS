#!/usr/bin/python
from sklearn.decomposition import PCA
import numpy as np
import scipy
import imp
try:
    imp.find_module('theano')
    use_theano = True
except ImportError:
    use_theano = False

if use_theano:
    from libSMACOF_theano import SMACOF
else:
    from libSMACOF import SMACOF

#Principal component analysis
def pca(data, n_components = 2):
    pca = PCA(n_components= n_components)
    return (pca.fit(data).transform(data))

def mds(data):
    smacof= SMACOF(data, 2)
    return smacof.solve(data)

