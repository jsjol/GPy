# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

# Kurt Cutajar, Jens Sjölund

import unittest
import numpy as np
import GPy
from GPy.inference.latent_function_inference.gaussian_grid_inference \
    import (_expand, _get_factorized_kernel, factor_grid)


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


class FactorizationTest(unittest.TestCase):
    def setUp(self):
        self.kernel = (GPy.kern.RBF(input_dim=2, active_dims=[0, 1]) *
                       GPy.kern.RBF(input_dim=1, active_dims=[2]) *
                       GPy.kern.Exponential(input_dim=2, active_dims=[3, 4]))
        self.grid_dims = [[0], [1], [2, 3, 4]]

    def test_expand(self):
        kern_expanded = _expand(self.kernel)
        active_dims_expanded = [kern_expanded.parts[d].active_dims.tolist()
                                for d in range(len(kern_expanded.parts))]

        expected_active_dims_expanded = [[0], [1], [2], [3, 4]]

        np.testing.assert_array_equal(active_dims_expanded,
                                      expected_active_dims_expanded)

    def test_get_factorized_kernel(self):
        kern_factored = _get_factorized_kernel(_expand(self.kernel),
                                               self.grid_dims)
        active_dims_factored = [kern_factored[d].active_dims.tolist()
                                for d in range(len(kern_factored))]
        expected_active_dims_factored = [[0], [0], [0, 1, 2]]
        np.testing.assert_array_equal(active_dims_factored,
                                      expected_active_dims_factored)


class FactorGridTest(unittest.TestCase):
    def setUp(self):
        x = np.arange(2).reshape(2, 1)
        y = np.arange(4).reshape(2, 2)
        self.xg = [x, y]
        self.grid_dims = [[0], [1, 2]]
        self.X = np.array([[0, 0, 1],
                           [0, 2, 3],
                           [1, 0, 1],
                           [1, 2, 3]])

    def test_factor_grid(self):
        resulting_xg = factor_grid(self.X, self.grid_dims)
        np.testing.assert_(len(resulting_xg) == 2)
        np.testing.assert_array_equal(resulting_xg[0], self.xg[0])
        np.testing.assert_array_equal(resulting_xg[1], self.xg[1])
