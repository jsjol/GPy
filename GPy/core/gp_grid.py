# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

# Kurt Cutajar

#This implementation of converting GPs to state space models is based on the article:

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
import scipy.linalg as sp
from .gp import GP
from .parameterization.param import Param
from ..inference.latent_function_inference import gaussian_grid_inference
from .. import likelihoods
from ..kern import Prod, RBF
from ..util.linalg import (sequential_tensor_dot, kron_mmprod)

import logging
from GPy.inference.latent_function_inference.posterior import Posterior
logger = logging.getLogger("gp grid")

class GpGrid(GP):
    """
    A GP model for Grid inputs

    :param X: inputs
    :type X: np.ndarray (num_data x input_dim)
    :param likelihood: a likelihood instance, containing the observed data
    :type likelihood: GPy.likelihood.(Gaussian | EP | Laplace)
    :param kernel: the kernel (covariance function). See link kernels
    :type kernel: a GPy.kern.kern instance

    """

    def __init__(self, X, Y, kernel, likelihood, inference_method=None,
                 name='gp grid', Y_metadata=None, normalizer=False,
                 grid_dims=None):

        kernel = _expand(kernel.copy())

        inference_method = gaussian_grid_inference.GaussianGridInference(grid_dims)

        GP.__init__(self, X, Y, kernel, likelihood, inference_method=inference_method, name=name, Y_metadata=Y_metadata, normalizer=normalizer)
        self.posterior = None

    def parameters_changed(self):
        """
        Method that is called upon any changes to :class:`~GPy.core.parameterization.param.Param` variables within the model.
        In particular in the GP class this method reperforms inference, recalculating the posterior and log marginal likelihood and gradients of the model

        .. warning::
            This method is not designed to be called manually, the framework is set up to automatically call this method upon changes to parameters, if you call
            this method yourself, there may be unexpected consequences.
        """
        self.posterior, self._log_marginal_likelihood, self.grad_dict = self.inference_method.inference(self.kern, self.X, self.likelihood, self.Y_normalized, self.Y_metadata)
        self.likelihood.update_gradients(self.grad_dict['dL_dthetaL'])
        self.kern.update_gradients_direct(self.grad_dict['dL_dParams'])

    def _raw_predict(self, Xnew, full_cov=False, kern=None, compute_var=True):
        """
        Make a prediction for the latent function values
        """
        if kern is None:
            kern = self.kern

        # Define aliases for convenience
        noise = self.likelihood.variance
        Qs = self.posterior.Qs
        V_kron = self.posterior.V_kron

        # n = X.shape[0] (number of training samples)
        # m = Xnew.shape[0] (number of test samples)

        if isinstance(self.X, list) and isinstance(Xnew, list):
            if not (len(self.X) == len(self.inference_method.grid_dims) and
                    len(Xnew) == len(self.inference_method.grid_dims)):
                raise Exception("When specifying X and Xnew as axes, \
                                their dimensions need to match \
                                those of grid_dims.")

            # TODO: reduce memory use by looping over m - use generator?
            Knm, _ = self.inference_method.get_separate_covariances(
                        kern,
                        xg=self.X,
                        xg2=Xnew)

            mu = sequential_tensor_dot([K.T for K in Knm],
                                       self.posterior.alpha)
            mu = mu.reshape(-1, 1)

            if not compute_var:
                return mu

            A = (np.dot(Qs[i].T, Knm[i]) for i in range(len(Knm)))
            A = reduce(np.kron, A)

            D = np.sqrt(1 / (V_kron + noise))
            A = D * A

            Kmm, _ = self.inference_method.get_separate_covariances(
                        kern,
                        xg=Xnew)
            Kmm = reduce(np.kron, Kmm)

            var = np.diag(Kmm) - np.diag(np.dot(A.T, A))
            var = var.reshape(-1, 1)
        else:
            # compute mean predictions
            Knm = kern.K(self.X, Xnew)
            mu = np.dot(Knm.T, self.posterior.alpha)
            mu = mu.reshape(-1, 1)

            if not compute_var:
                return mu

            # compute variance of predictions
            A = kron_mmprod([Q.T for Q in Qs], Knm)
            A = A / (V_kron + noise)
            A = kron_mmprod(Qs, A)

            Kmm = kern.K(Xnew)
            var = np.diag(Kmm - np.dot(Knm.T, A)).copy()
            var = var.reshape(-1, 1)

        return mu, var


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
