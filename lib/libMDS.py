#!/usr/bin/python
from sklearn.decomposition import PCA
import imp

#Use Theano only if available
try:
    imp.find_module('theano')
    use_theano = True
except ImportError:
    use_theano = False

#use SMACOF from libSMACOF_theano when Theano avalailable
#from numpy implemention from libSMACOF when not
if use_theano:
    from .libSMACOF_theano import SMACOF
    print("Using Theano")
else:
    from .libSMACOF import SMACOF


#Principal component analysis
def pca(data, n_components=2):
    """
    PCA Transform

    Parameters
    ---------
    data: array-like
        initial high dimensional data
    n_components, optional, default: 2
        number of dimensions data should be transformed to

    Returns
    -------
    pca_data: array-like
        PCA transformed low dimensional data
        """

    pca = PCA(n_components=n_components)
    return (pca.fit(data).transform(data))


def mds(data):
    """
    Multidimensional Scaling using SMACOF

    Parameters
    ---------
    data: array-like
        initial high dimensional data
    n_components, optional, default: 2
        number of dimensions data should be transformed to

    Returns
    -------
    mds_data: array-like
        PCA transformed low dimensional data
        """
    smacof = SMACOF(data, n_components=2)
    mds_data = smacof.solve(data)
    return mds_data
