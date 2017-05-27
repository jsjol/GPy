# Copyright (c) 2017, Jens Sjölund

import warnings
import numpy as np
import numpy.polynomial.legendre as L
from .kern import Kern
from ...core.parameterization import Param
from paramz.transformations import Logexp
from paramz.caching import Cache_this


class LegendrePolynomial(Kern):
    """
    Legendre polynomial kernel:

    .. math::

       k(x, y) = \sum_{n\in \text{orders}} c_n P_n(x^T y) \\ \\ \\ \\ \
       where P_n is the Legendre polynomial of order n.}

    The Legendre polynomials with non-negative coefficients form a basis for
    valid covariance functions on the sphere, provided that x and y are vectors
    on the unit sphere. Normalization of the inputs is, however, not done
    by this function.

    :param coefficients: coefficients for the nonzero terms specified by orders
    :type coefficients: array-like
    :param orders: the orders n of the nonzero terms in the sum
    :type orders: array-like
    """

    def __init__(self, input_dim, coefficients=(1., 1.),
                 orders=(0, 2), active_dims=None, name='LegendrePolynomial'):
        super(LegendrePolynomial, self).__init__(input_dim, active_dims, name)

        assert np.min(orders) >= 0, \
            'The order of the polynomial has to be at least 0.'

        if not len(coefficients) == len(orders):
            warnings.warn('The number of coefficents does not agree with the\
                          number of orders. Resetting all coefficients to 1.')
            coefficients = np.ones(len(orders))

        self.coefficients = Param('coefficients', coefficients, Logexp())
        self.link_parameters(self.coefficients)
        self.orders = orders

    def K(self, X, X2=None):
        V, c = self._getVandermondeMatrixAndCoefficients(X, X2)
        return np.dot(V, c)

    @Cache_this(limit=3)
    def _getVandermondeMatrixAndCoefficients(self, X, X2=None):
        """
        Compute and cache the pseudo-Vandermonde matrix V and inflate the
        coefficients into a 1-D array c of length n + 1. The Legendre
        polynomial can then be evaluated as np.dot(V, c).
        """
        if X2 is None:
            dot_prod = np.dot(X, X.T)
        else:
            dot_prod = np.dot(X, X2.T)
        highestOrder = np.max(self.orders)
        V = L.legvander(dot_prod, highestOrder)
        c = np.zeros(highestOrder + 1)
        c[np.array(self.orders)] = self.coefficients.values
        return V, c

    @Cache_this(limit=3)
    def dK_dParams(self, X, X2=None):
        V, _ = self._getVandermondeMatrixAndCoefficients(X, X2)
        V = V[:, :, self.orders]
        return V

    def Kdiag(self, X):
        return self.K(X).diagonal()

    def update_gradients_full(self, dL_dK, X, X2=None):
        dK_dCoefficients = self.dK_dParams(X, X2)
        dL_dCoefficients = dL_dK[:, :, np.newaxis] * dK_dCoefficients
        self.coefficients.gradient = np.sum(dL_dCoefficients, axis=(0, 1))

    def update_gradients_direct(self, dL_dCoefficients):
        """
        Specially intended for the Grid regression case.
        """
        self.coefficients.gradient = dL_dCoefficients

    def update_gradients_diag(self, dL_dKdiag, X):
        raise NotImplementedError

    def gradients_X(self, dL_dK, X, X2=None):
        raise NotImplementedError

    def gradients_X_diag(self, dL_dKdiag, X):
        raise NotImplementedError
