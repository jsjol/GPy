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
    def __init__(self):
        pass

    def kron_mvprod(self, A, b):
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

    def inference(self, kern, X, likelihood, Y, Y_metadata=None):

        """
        Returns a GridPosterior class containing essential quantities of the posterior
        """

        if isinstance(kern, RBF):
            # Convert the RBF kernel to a product of 1D kernels
            kern = kern.as_product_kernel()
        
        assert isinstance(kern, Prod), \
            "Grid inference only implemented for RBF or product kernel."

        N = X.shape[0] #number of training points
        D = kern.num_params #number of kernel factors

        xg = factor_grid(X) # TODO: make smarter than assuming 1D

        Kds = np.zeros(D, dtype=object) #vector for holding covariance per dimension
        Qs = np.zeros(D, dtype=object) #vector for holding eigenvectors of covariance per dimension
        QTs = np.zeros(D, dtype=object) #vector for holding transposed eigenvectors of covariance per dimension
        V_kron = 1 # kronecker product of eigenvalues

        # This follows algorithm 16 in Saatci (2011)
        
        # If you could automatically retrieve the hyperparameters of each part
        # in the (product) kernel, then you could iterate over them. Hopefully
        # you would also be able to update the gradients of the parameters directly.
        
        # My current aim is to only make it sufficiently smart to parse if it
        # is a product kernel and then check whether each factor is rbf or 
        # Legendre.           

        for d in range(D):  # Saatci 1-4
            Kds[d] = kern.parts[d].K(xg[d])
            [V, Q] = np.linalg.eig(Kds[d])
            V_kron = np.kron(V_kron, V)
            Qs[d] = Q
            QTs[d] = Q.T

        noise = likelihood.variance + 1e-8

        alpha_kron = self.kron_mvprod(QTs, Y)  # Saatci 5
        V_kron = V_kron.reshape(-1, 1)
        alpha_kron = alpha_kron / (V_kron + noise)  # Saatci 6
        alpha_kron = self.kron_mvprod(Qs, alpha_kron)  # Saatci 7

        posterior = GridPosterior(alpha_kron=alpha_kron, QTs=QTs,
                                  Qs=Qs, V_kron=V_kron, noise=noise)

        log_likelihood = -0.5 * (np.dot(Y.T, posterior.alpha) +
                                 np.sum((np.log(posterior.V_kron +
                                                posterior.noise))) +
                                 N*np.log(2*np.pi))  # Saatci 8

        gradient_dict = self.compute_gradients(posterior, kern, xg, Kds)

        return (posterior, log_likelihood, gradient_dict)

    def compute_gradients(self, posterior, kern, xg, Kds):

        D = kern.num_params

        # Saatci 11 (extended)
        dKd_dTheta = [kern.parts[d].dK_dtheta(xg[d]) for d in range(D)]

        # Loop over theta i (flattened) (could be replaced by a double loop)
            # gam = 1 (cumulative)
            # Loop over kernels d
                # If theta i belongs to kernel d (this if statement corresponds to the cryptic t==d argument)
                    # gamma_d = diag(Q'*dK_dtheta_i*Q)
                    # kappa_array[d] = dK_dtheta_i
                # else
                    # gamma_d = diag(Q'*K*Q)
                    # kappa_array[d] = K
                # gam = kron(gam, gamma_d)
            # kappa = mvprod(kappa_array, alpha)        

        gradients = np.array([])
        for i in range(D):  # Double loops over theta = theta_ij
            for j, _ in enumerate(kern.parts[i].param_array):

                kappa_parts = []
                for d in range(D):  # Loop over kernels
                    if i == d:
                        kappa_parts.append(dKd_dTheta[i][:, :, j])
                    else:
                        kappa_parts.append(Kds[d])
                kappa = self.kron_mvprod(kappa_parts, posterior.alpha)

                gamma = compute_gamma(posterior, kappa_parts)

                gradient_ij = dL_dTheta(posterior, kappa, gamma)
                gradients = np.append(gradients, gradient_ij)

        # Noise variance gradient
        kappa_parts = [np.identity(xg[d].shape[0]) for d in range(D)]
        kappa = self.kron_mvprod(kappa_parts, posterior.alpha)

        gamma = compute_gamma(posterior, kappa_parts)

        dL_dNoise_variance = dL_dTheta(posterior, kappa, gamma)

#        
#        derivs = np.zeros(D+2, dtype='object')
#        for t in range(len(derivs)): # Saatci 9: loop over theta with index i
#            dKd_dTheta = np.zeros(D, dtype='object')
#            gam = 1
#            for d in range(D):
#                
#                if t < D:
#                    dKd_dTheta[d] = kern.parts[d].dKd_dLen(xg[d], (t==d), lengthscale=kern.lengthscale[t]) #derivative wrt lengthscale
#                elif (t == D):
#                    dKd_dTheta[d] = kern.parts[d].dKd_dVar(xg[d]) #derivative wrt variance
#                else:
#                    dKd_dTheta[d] = np.identity(len(xg[d])) #derivative wrt noise
#                gamma_d = np.diag(np.dot(np.dot(QTs[d], dKd_dTheta[d].T), Qs[d])) # Probably not the most efficient since it first forms the full matrix and then picks out the diagonal
#                gam = np.kron(gam, gamma_d) # Could use reduce on a list of of gamma_d's instead
#            
#            gam = gam.reshape(-1,1)
#            kappa = self.kron_mvprod(dKd_dTheta, alpha_kron) # Saatci 15
#            derivs[t] = 0.5*np.dot(alpha_kron.T,kappa) - 0.5*np.sum(gam / (V_kron + noise)) # Saatci 16
        
        is_lengthscale = ['lengthscale' in param_name for param_name in kern.parameter_names()]
        is_variance = ['variance' in param_name for param_name in kern.parameter_names()]

        dL_dLen = gradients[np.nonzero(is_lengthscale)]

        dL_dVar_part = gradients[np.nonzero(is_variance)]
        var_parts = kern.param_array[np.nonzero(is_variance)]
        dVar_part_dVar_product = 1./D * var_parts ** (1 - D)
        dL_dVar = np.dot(dL_dVar_part, dVar_part_dVar_product)

        return {'dL_dLen': dL_dLen,
                'dL_dVar': dL_dVar,
                'dL_dthetaL': dL_dNoise_variance}


def factor_grid(X):
    """
       Extract the unique values for each dimension
    """
    xg = []
    D = X.shape[1]
    for d in range(D):
        unique_elements = np.unique(X[:, d])
        xg.append(unique_elements[:, np.newaxis])

    return xg


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
