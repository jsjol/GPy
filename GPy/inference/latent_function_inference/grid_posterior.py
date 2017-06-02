# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np

class GridPosterior(object):
    """
    Specially intended for the Grid Regression case
    An object to represent a Gaussian posterior over latent function values, p(f|D).

    The purpose of this class is to serve as an interface between the inference
    schemes and the model classes.

    """
    def __init__(self, alpha_kron, Qs, V_kron, noise=None):
        """
        alpha_kron : 
        Qs : eigen vectors resulting from decomposition of single dimension covariance matrices
        V_kron : kronecker product of eigenvalues reulting decomposition of single dimension covariance matrices
        """

        self._alpha_kron = alpha_kron
        self._qs = Qs
        self._v_kron = V_kron
        self._noise = noise

    @property
    def alpha(self):
        """
        """
        return self._alpha_kron

    @property
    def Qs(self):
        """
        array of eigenvectors resulting for single dimension covariance
        """
        return self._qs

    @property
    def V_kron(self):
        """
        kronecker product of eigenvalues s
        """
        return self._v_kron

    @property
    def noise(self):
        """
        kronecker product of eigenvalues s
        """
        return self._noise
    
