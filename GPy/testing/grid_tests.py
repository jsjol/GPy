# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

# Kurt Cutajar

import unittest
import numpy as np
import GPy


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

        # import pdb; pdb.set_trace()
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
