# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

# Kurt Cutajar

# This implementation of converting GPs to state space models is based on the article:

#@article{Gilboa:2015,
#  title={Scaling multidimensional inference for structured Gaussian processes},
#  author={Gilboa, Elad and Saat{\c{c}}i, Yunus and Cunningham, John P},
#  journal={Pattern Analysis and Machine Intelligence, IEEE Transactions on},
#  volume={37},
#  number={2},
    #  pages={424--436},
#  year={2015},
#  publisher={IEEE}
#}

from functools import reduce
from itertools import chain
import numpy as np
from .grid_posterior import GridPosterior
from ...kern import Prod, RBF
from . import LatentFunctionInference


class GaussianGridInference(LatentFunctionInference):
    """
    An object for inference when the likelihood is Gaussian and inputs are on a grid.

    The function self.inference returns a GridPosterior object, which summarizes
    the posterior.

    """
    def __init__(self, grid_dims):
        self.grid_dims = grid_dims

    def inference(self, kern, X, likelihood, Y, Y_metadata=None):

        """
        Returns a GridPosterior class containing essential quantities of the posterior
        """
        assert (isinstance(kern, RBF) or isinstance(kern, Prod)), \
            "Grid inference only implemented for RBF and product kernel."

        kern = _expand(kern)

        if self.grid_dims is None:
            self.grid_dims = [[d] for d in range(X.shape[1])]

        N = X.shape[0] #number of training points

        xg = factor_grid(X, self.grid_dims) # TODO: make smarter than assuming 1D
        kern_factors = _get_factorized_kernel(kern, self.grid_dims)

        D = len(kern_factors)
        Kds = [kern_factors[d].K(xg[d]) for d in range(D)]

        Qs = np.zeros(D, dtype=object) #vector for holding eigenvectors of covariance per dimension
        QTs = np.zeros(D, dtype=object) #vector for holding transposed eigenvectors of covariance per dimension
        V_kron = 1 # kronecker product of eigenvalues

        # This follows algorithm 16 in Saatci (2011)
        
        # My current aim is to only make it sufficiently smart to parse if it
        # is a product kernel and then check whether each factor is rbf or 
        # Legendre.           

        for d in range(D):  # Saatci 1-4
            [V, Q] = np.linalg.eigh(Kds[d])
            V_kron = np.kron(V_kron, V)
            Qs[d] = Q
            QTs[d] = Q.T

        noise = likelihood.variance + 1e-8

        alpha_kron = kron_mvprod(QTs, Y)  # Saatci 5 # TODO: Remove the need for stored QTs
        V_kron = V_kron.reshape(-1, 1)
        alpha_kron = alpha_kron / (V_kron + noise)  # Saatci 6
        alpha_kron = kron_mvprod(Qs, alpha_kron)  # Saatci 7

        posterior = GridPosterior(alpha_kron=alpha_kron, Qs=Qs,
                                  V_kron=V_kron, noise=noise)

        log_likelihood = -0.5 * (np.dot(Y.T, posterior.alpha) +
                                 np.sum((np.log(posterior.V_kron +
                                                posterior.noise))) +
                                 N*np.log(2*np.pi))  # Saatci 8

        gradient_dict = self.compute_gradients(posterior, kern, kern_factors,
                                               xg, Kds)

        return (posterior, log_likelihood, gradient_dict)

    def compute_gradients(self, posterior, kern, kern_factors, xg, Kds):
        D = len(kern_factors)

        dKd_dTheta = [kern_factors[d].dK_dtheta(xg[d]) for d in range(D)]      

        gradients = np.array([])
        for i in range(D):  # Double loops over theta = theta_ij
            for j, _ in enumerate(kern_factors[i].param_array):

                kappa_parts = []
                for d in range(D):  # Loop over kernels
                    if i == d:
                        kappa_parts.append(dKd_dTheta[i][:, :, j])
                    else:
                        kappa_parts.append(Kds[d])
                kappa = kron_mvprod(kappa_parts, posterior.alpha)

                gamma = compute_gamma(posterior, kappa_parts)

                gradient_ij = dL_dTheta(posterior, kappa, gamma)
                gradients = np.append(gradients, gradient_ij)

        # Noise variance gradient ---------------
        kappa_parts = [np.identity(xg[d].shape[0]) for d in range(D)]
        kappa = kron_mvprod(kappa_parts, posterior.alpha)

        gamma = compute_gamma(posterior, kappa_parts)

        dL_dNoise_variance = dL_dTheta(posterior, kappa, gamma)
        # -----------------------------------------

        is_lengthscale = ['lengthscale' in param_name
                          for param_name in kern.parameter_names()]
        is_variance = ['variance' in param_name
                       for param_name in kern.parameter_names()]

        dL_dLen = gradients[np.nonzero(is_lengthscale)]

        dL_dVar_part = gradients[np.nonzero(is_variance)]
        var_parts = kern.param_array[np.nonzero(is_variance)]
        dVar_part_dVar_product = 1./D * var_parts ** (1 - D)
        dL_dVar = np.dot(dL_dVar_part, dVar_part_dVar_product)

        return {'dL_dLen': dL_dLen,
                'dL_dVar': dL_dVar,
                'dL_dthetaL': dL_dNoise_variance}


def _expand(kern):
    """
    Recursively expand parts that are either RBFs or products themselves
    """
    if isinstance(kern, RBF):
        return kern.as_product_kernel()
    elif isinstance(kern, Prod):
        children = [_expand(kern.parts[d]) for d in range(len(kern.parts))]
        kern_expanded = reduce((lambda child_1, child_2: child_1 * child_2), children)
        return kern_expanded
    else:
        return kern


def _get_factorized_kernel(kern, grid_dims):
    kern_factored = []
    leftovers = []
    for i in range(len(kern.parts)):
        k = kern.parts[i].copy()

        if list_in_list(kern.parts[i].active_dims.tolist(), grid_dims):
            _append(kern_factored, k)
        else:
            if not leftovers:
                leftovers = k
            else:
                leftovers *= k

    if leftovers:
        _append(kern_factored, leftovers)

    return kern_factored


def list_in_list(element, list_to_search):
    comparison = [element == el for el in list_to_search]
    return np.any(comparison)


def _append(kern_factored, kern_to_append):
    kern_to_append = _set_active_dims_to_none(kern_to_append)
    return kern_factored.append(kern_to_append)


def _set_active_dims_to_none(k):
    # For correct behavior when passing pre-sliced Xs
    # TODO: handle the details elsewhere (part of Kern.__init__)
    k.active_dims = np.arange(k.input_dim, dtype=int)
    k._all_dims_active = k.active_dims
    return k


#def _detect_factorization(kern):
#    """
#    Recursively detect a valid factorization
#    """
#    if isinstance(kern, RBF):
#        return [[kern.active_dims[d]] for d in range(kern.input_dim)]
#    elif isinstance(kern, Prod):
#        L = len(kern.parts)
#        children = [_detect_factorization(kern.parts[d]) for d in range(L)]
#        return list(chain.from_iterable(children))
#    else:
#        return [list(kern.active_dims)]


def factor_grid(X, grid_dims):
    """
       Extract the unique values for each dimension
    """
    xg = []
    for d in grid_dims:
        unique_elements = unique_rows(X[:, d])
        if unique_elements.ndim < 2:
            unique_elements = unique_elements[:, np.newaxis]
        xg.append(unique_elements)
    return xg


def unique_rows(X):
    """
    Copied from http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array
    This feature will come natively in numpy 1.13
    """
    X_view = np.ascontiguousarray(X).view(np.dtype((np.void, X.dtype.itemsize * X.shape[1])))
    _, idx = np.unique(X_view, return_index=True)
    return X[idx]


def kron_mvprod(A, b):
    """
        Perform the matrix-vector multiplication A*b where the matrix A is
        given by a Kronecker product.
    """
    x = b
    N = 1
    D = len(A)
    G = np.zeros((D, 1))
    for d in range(0, D):
        G[d] = len(A[d])
    N = np.prod(G)
    for d in range(D-1, -1, -1):
        X = np.reshape(x, (G[d], np.round(N/G[d])), order='F')
        Z = np.dot(A[d], X)
        Z = Z.T
        x = np.reshape(Z, (-1, 1), order='F')
    return x


def compute_gamma(posterior, kappa_parts):
    D = len(kappa_parts)
    gamma_d = [_compute_gamma_d(posterior.QTs[d],
                                kappa_parts[d],
                                posterior.Qs[d])
               for d in range(D)]
    gamma = reduce(np.kron, gamma_d)
    return gamma.reshape(-1, 1)


def _compute_gamma_d(P_T, A, P):
    # This could be made more efficiently using np.einsum()
    # Note: K transposed (typo in Saatci eq. (5.36), but this
    # doesn't matter since it's symmetric)
    return np.diag(np.dot(np.dot(P_T, A.T), P))


def dL_dTheta(posterior, kappa, gamma):
    return (0.5*np.dot(posterior.alpha.T, kappa) -
            0.5*np.sum(gamma / (posterior.V_kron + posterior.noise)))
