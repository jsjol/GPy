# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from functools import reduce


class GridPosterior(object):
    """
    Specially intended for the Grid Regression case
    An object to represent a Gaussian posterior over
    latent function values, p(f|D).

    The purpose of this class is to serve as an interface between the inference
    schemes and the model classes.

    """
    def __init__(self, alpha_kron, Qs, Vs, V_kron=None, noise=None):
        """
        alpha_kron : woodbury vector
        Qs : list of eigenvector matrices resulting from separate decomposition
             of covariance matrices in a Kronecker product
        V : corresponding eigenvalues
        """

        self._alpha_kron = alpha_kron
        self._qs = Qs
        self._vs = Vs
        self._v_kron = V_kron
        self._noise = noise

    @property
    def alpha(self):
        return self._alpha_kron

    @property
    def Qs(self):
        """
        list of eigenvectors
        """
        return self._qs

    @property
    def Vs(self):
        """
        list of eigenvalues
        """
        return self._vs

    @property
    def V_kron(self):
        """
        kronecker product of eigenvalues
        """
        if self._v_kron is None:
            self._v_kron = reduce(np.kron, self._vs).reshape(-1, 1)
        return self._v_kron

    @property
    def noise(self):
        return self._noise
