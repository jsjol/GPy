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

    This follows algorithm 16 in Saatci (2011)

    """
    def __init__(self, grid_dims):
        self.grid_dims = grid_dims

    def inference(self, kern, X, likelihood, Y, Y_metadata=None):

        """
        Returns a GridPosterior class containing essential quantities of the posterior
        """
#        assert (isinstance(kern, RBF) or isinstance(kern, Prod)), \
#            "Grid inference only implemented for RBF and product kernel."
        # TODO: update checks above. The case when len(grid_dims) == 1 does not 
        # not benefit from Grid inference, but could be useful for testing

#        if self.grid_dims is None:
#            self.grid_dims = [[d] for d in range(X.shape[1])]

        if isinstance(X, list):
            if self.grid_dims is None:
                raise Exception("Can't infer grid_dims, please specify.")
            xg = X
        else:
            if self.grid_dims is None:
                self.grid_dims = [[d] for d in range(X.shape[1])]
            xg = factor_grid(X, self.grid_dims)

        Kds, kern_factors = self.get_separate_covariances(kern, xg)

        posterior = self.compute_posterior(likelihood, Kds, Y)

        log_likelihood = -0.5 * (np.dot(Y.T, posterior.alpha) +
                                 np.sum((np.log(posterior.V_kron +
                                                posterior.noise))) +
                                 Y.shape[0]*np.log(2*np.pi))  # Saatci 8

        gradient_dict = self.compute_gradients(posterior, kern,
                                               kern_factors, xg, Kds)

        return (posterior, log_likelihood, gradient_dict)

    def get_separate_covariances(self, kern, xg, xg2=None):
        kern_factors = _get_factorized_kernel(kern, self.grid_dims)

        if xg2 is None:
            xg2 = xg

        Kds = [kern_factors[d].K(xg[d], xg2[d])
               for d in range(len(kern_factors))]

        return Kds, kern_factors

    def compute_posterior(self, likelihood, Kds, Y):
        Vs = []
        Qs = []
        for K in Kds:  # Saatci 1-4
            [V, Q] = np.linalg.eigh(K)
            Vs.append(V)
            Qs.append(Q)

        noise = likelihood.variance + 1e-8
        V_kron = reduce(np.kron, Vs).reshape(-1, 1)

        alpha_kron = kron_mvprod([Q.T for Q in Qs], Y)  # Saatci 5 #
        alpha_kron = alpha_kron / (V_kron + noise)  # Saatci 6
        alpha_kron = kron_mvprod(Qs, alpha_kron)  # Saatci 7

        return GridPosterior(alpha_kron=alpha_kron, Qs=Qs, Vs=Vs,
                             V_kron=V_kron, noise=noise)

    def compute_gradients(self, posterior, kern, kern_factors, xg, Kds):
        D = len(kern_factors)

        dKd_dParams = [kern_factors[d].dK_dParams(xg[d]) for d in range(D)]

        dL_dParams = []
        for i, factor in enumerate(kern_factors):  # Double loops over theta = theta_ij
            dL_dFactorParams = np.array([])
            for j, _ in enumerate(factor.param_array):
                kappa_factors = []
                for d in range(D):  # Loop over kernels
                    if i == d:
                        kappa_factors.append(dKd_dParams[i][:, :, j])
                    else:
                        kappa_factors.append(Kds[d])

                dL_dFactorParams = np.append(
                    dL_dFactorParams,
                    _dL_dParam_from_parts(kappa_factors, posterior))

            dL_dPartParams = _expand_prod_gradient(factor, dL_dFactorParams)
            # The gradient update requires the gradients of each part
            # in a product kernel separately
            for g in dL_dPartParams:
                dL_dParams.append(g)

        if len(dL_dParams) == 1:
            dL_dParams = dL_dParams[0]

        # Noise variance gradient -----------------
        kappa_factors = [np.identity(xg[d].shape[0]) for d in range(D)]
        dL_dNoise_variance = _dL_dParam_from_parts(kappa_factors, posterior)
        # -----------------------------------------

        return {'dL_dParams': dL_dParams,
                'dL_dthetaL': dL_dNoise_variance}


def _expand_prod_gradient(kern, flattened_dL_dParams):
    if isinstance(kern, Prod):
        idx = 0
        out = []
        for part in kern.parts:
            n = len(part.param_array)
            out.append(flattened_dL_dParams[idx:idx + n])
            idx += n
    else:
        out = [flattened_dL_dParams]

    return out


def _get_factorized_kernel(kern, grid_dims):
    if len(grid_dims) == 1:
        return [kern]

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
    kern_to_append = _update_sliced_active_dims(kern_to_append)
    return kern_factored.append(kern_to_append)


def _update_sliced_active_dims(k, shift=None):
    if k.active_dims is None:
        k.active_dims = np.arange(k.input_dim, dtype=int)
    else:
        if shift is None:
            shift = k.active_dims[0]
        k.active_dims = np.atleast_1d(k.active_dims) - shift

    k._all_dims_active = k.active_dims

    if isinstance(k, Prod):
        for i, kern in enumerate(k.parts):
            k.parts[i] = _update_sliced_active_dims(k.parts[i], shift=shift)

    return k


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
    return X[sorted(idx)]  # Sorting necessary to preserve ordering


def sequential_tensor_dot(A_seq, B):
    B = np.reshape(B, [A.shape[1] for A in A_seq])
    for A in reversed(A_seq):
        B = np.tensordot(A, B, axes=(1, -1))
    return B


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


def _dL_dParam_from_parts(kappa_factors, posterior):
    kappa = kron_mvprod(kappa_factors, posterior.alpha)
    gamma = compute_gamma(posterior, kappa_factors)
    return compute_dL_dParams(posterior, kappa, gamma)


def compute_gamma(posterior, kappa_factors):
    D = len(kappa_factors)
    gamma_d = [_compute_gamma_d(kappa_factors[d],
                                posterior.Qs[d])
               for d in range(D)]
    gamma = reduce(np.kron, gamma_d)
    return gamma.reshape(-1, 1)


def _compute_gamma_d(A, Q):
    # This could be made more efficiently using np.einsum()
    return np.diag(np.dot(np.dot(A, Q).T, Q))


def compute_dL_dParams(posterior, kappa, gamma):
    return (0.5*np.dot(posterior.alpha.T, kappa) -
            0.5*np.dot(gamma.T,  1/(posterior.V_kron + posterior.noise)))
