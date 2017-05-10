# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

# Kurt Cutajar

import unittest
import numpy as np
import GPy
from GPy.inference.latent_function_inference.gaussian_grid_inference \
    import (_expand, _get_factorized_kernel)


class GridModelTest(unittest.TestCase):
    def setUp(self):
        ######################################
        # # 3 dimensional example

        # sample inputs and outputs
        np.random.seed(0)
        self.X = np.array([[0, 0, 0],
                           [0, 0, 1],
                           [0, 1, 0],
                           [0, 1, 1],
                           [1, 0, 0],
                           [1, 0, 1],
                           [1, 1, 0],
                           [1, 1, 1]])
        self.Y = np.random.randn(8, 1) * 100
        self.dim = self.X.shape[1]

        self.kernel = GPy.kern.RBF(input_dim=self.dim, lengthscale=(2, 3, 4),
                                   variance=2, ARD=True)
        self.m = GPy.models.GPRegressionGrid(self.X, self.Y, self.kernel)

        self.kernel2 = GPy.kern.RBF(input_dim=self.dim, lengthscale=(2, 3, 4),
                                    variance=2, ARD=True)
        self.m2 = GPy.models.GPRegression(self.X, self.Y, self.kernel2)

    def test_alpha_match(self):
        np.testing.assert_almost_equal(self.m.posterior.alpha,
                                       self.m2.posterior.woodbury_vector)

    def test_gradient_match(self):
        np.testing.assert_almost_equal(self.kernel.lengthscale.gradient,
                                       self.kernel2.lengthscale.gradient)
        np.testing.assert_almost_equal(self.m.likelihood.variance.gradient,
                                       self.m2.likelihood.variance.gradient)
        np.testing.assert_almost_equal(self.kernel.variance.gradient,
                                       self.kernel2.variance.gradient)

    def test_prediction_match(self):
        test = np.array([[0, 0, 2], [-1, 3, -4]])

        np.testing.assert_almost_equal(self.m.predict(test),
                                       self.m2.predict(test))


class AutoFactorizationTest(unittest.TestCase):
    def setUp(self):
        self.kernel = (GPy.kern.RBF(input_dim=2, active_dims=[0, 1]) *
                       GPy.kern.RBF(input_dim=1, active_dims=[2]) *
                       GPy.kern.Exponential(input_dim=2, active_dims=[3, 4]))
        self.grid_dims = [[0], [1], [2, 3, 4]]

#    def test_detect_factorization(self):
#        active_dims_expanded = _detect_factorization(self.kernel)
#        #import pdb; pdb.set_trace()
#        #n_parts_expanded = len(kern_expanded.parts)
#        #np.testing.assert_equal(n_parts_expanded, 5)
#
#        #active_dims_expanded = [kern_expanded.parts[d].active_dims
#         #                       for d in range(n_parts_expanded)]
#        #import pdb; pdb.set_trace()
#        expected_active_dims_expanded = [[0], [1], [1], [2], [3, 4]]
#        np.testing.assert_array_equal(active_dims_expanded,
#                                      expected_active_dims_expanded)
    def test_expand(self):
        kern_expanded = _expand(self.kernel)
        active_dims_expanded = [kern_expanded.parts[d].active_dims.tolist()
                                for d in range(len(kern_expanded.parts))]

        expected_active_dims_expanded = [[0], [1], [2], [3, 4]]

        np.testing.assert_array_equal(active_dims_expanded,
                                      expected_active_dims_expanded)

    def test_get_factorized_kernel(self):
        kern_factored = _get_factorized_kernel(self.kernel, self.grid_dims)
        active_dims_factored = [kern_factored[d].active_dims.tolist()
                                for d in range(len(kern_factored.parts))]
        expected_active_dims_factored = [[0], [1], [2, 3, 4]]
        np.testing.assert_array_equal(active_dims_factored,
                                      expected_active_dims_factored)
