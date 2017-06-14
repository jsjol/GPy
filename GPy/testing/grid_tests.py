# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import unittest
import numpy as np
import GPy
from GPy.inference.latent_function_inference.gaussian_grid_inference \
    import (_get_factorized_kernel, factor_grid, kron_mvprod,
            sequential_tensor_dot)
from GPy.core.gp_grid import (_expand, kron_mmprod)


class KroneckerTests(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.A = np.random.randn(3, 3)
        self.B = np.random.randn(4, 4)
        self.kronAB = np.kron(self.A, self.B)
        self.x = np.random.randn(self.A.shape[1], 1)
        self.y = np.random.randn(self.A.shape[1] * self.B.shape[1], 1)

    def test_ordinary_mvprod(self):
        expected_mv_result = np.dot(self.A, self.x)
        kron_mv_result = kron_mvprod([self.A], self.x)
        np.testing.assert_array_almost_equal(expected_mv_result,
                                             kron_mv_result)

    def test_kron_mvprod(self):
        expected = np.dot(self.kronAB, self.y)
        kron_mv_result = kron_mvprod([self.A, self.B], self.y)
        np.testing.assert_array_almost_equal(kron_mv_result,
                                             expected)

    def test_kron_mmprod(self):
        Y = np.random.randn(self.A.shape[1] * self.B.shape[1], 3)
        expected = np.dot(self.kronAB, Y)
        kron_mm_result = kron_mmprod([self.A, self.B], Y)
        np.testing.assert_almost_equal(kron_mm_result, expected)


class SequentialTensorDotTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.A = np.random.randn(2, 3)
        self.B = np.random.randn(4, 5)
        self.kronAB = np.kron(self.A, self.B)
        self.x = np.random.randn(self.A.shape[1])
        self.y = np.random.randn(self.A.shape[1], self.B.shape[1])

    def test_ordinary_mvprod(self):
        expected_mv_result = np.dot(self.A, self.x)
        tensor_dot_result = sequential_tensor_dot([self.A], self.x)
        np.testing.assert_array_almost_equal(tensor_dot_result,
                                             expected_mv_result)

    def test_kron_mvprod(self):
        expected = np.dot(self.kronAB, self.y.flatten())
        expected = np.reshape(expected, (self.A.shape[0], self.B.shape[0]))
        tensor_dot_result = sequential_tensor_dot([self.A, self.B], self.y)
        np.testing.assert_array_almost_equal(tensor_dot_result,
                                             expected)


class GridModelIdentityTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        n = 3
        self.X = np.arange(n).reshape((n, 1))
        self.Xs = np.array([[0.5], [0]])
        self.Y = np.random.randn(n, 1)
        self.dim = self.X.shape[1]
        self.grid_dims = [[0]]

        self.kernel = GPy.kern.RBF(input_dim=self.dim, lengthscale=2,
                                   variance=2, ARD=False)
        self.kernel2 = self.kernel.copy()

        self.m = GPy.models.GPRegressionGrid(self.X, self.Y, self.kernel,
                                             grid_dims=self.grid_dims)
        self.m2 = GPy.models.GPRegression(self.X, self.Y, self.kernel2)

    def test_K_match(self):
        K = self.m.kern.K(self.m.X)
        K2 = self.m2.kern.K(self.m2.X)
        np.testing.assert_almost_equal(K, K2)

    def test_crosscovariance_match(self):
        Ks = self.m.kern.K(self.Xs, self.X)
        Ks2 = self.m2.kern.K(self.Xs, self.X)
        np.testing.assert_almost_equal(Ks, Ks2)

    def test_dK_dParams(self):
        dK_dParams = self.m.kern.dK_dParams(self.X)
        dK_dVar = dK_dParams[:, :, 0]
        dK_dLen = dK_dParams[:, :, 1]

        kern2 = self.m2.kern
        dK_dVar2 = kern2.K(self.X) / kern2.variance
        dK_dr = kern2.dK_dr_via_X(self.X, None)
        r = kern2._scaled_dist(self.X)
        dK_dLen2 = dK_dr * (-r/kern2.lengthscale)

        np.testing.assert_almost_equal(dK_dVar, dK_dVar2)
        np.testing.assert_almost_equal(dK_dLen, dK_dLen2)

    def test_alpha_match(self):
        np.testing.assert_almost_equal(self.m.posterior.alpha,
                                       self.m2.posterior.woodbury_vector)

    def test_likelihood(self):
        np.testing.assert_almost_equal(self.m.likelihood.variance,
                                       self.m2.likelihood.variance)
        np.testing.assert_almost_equal(self.m.log_likelihood(),
                                       self.m2.log_likelihood())

    def test_gradient_match(self):
        np.testing.assert_almost_equal(self.m.kern.lengthscale.gradient,
                                       self.m2.kern.lengthscale.gradient)
        np.testing.assert_almost_equal(self.m.likelihood.variance.gradient,
                                       self.m2.likelihood.variance.gradient)
        np.testing.assert_almost_equal(self.m.kern.variance.gradient,
                                       self.m2.kern.variance.gradient)

    def test_prediction_match(self):
        np.testing.assert_almost_equal(self.m.predict(self.Xs),
                                       self.m2.predict(self.Xs))


class GridModelIdentityTest2D(GridModelIdentityTest):
    """ Repeat the tests in GridModelIdentityTest with a
        different setup."""
    def setUp(self):
        np.random.seed(0)
        self.X = np.array([[0, 0],
                           [0, 1],
                           [1, 0],
                           [1, 1],
                           [2, 0],
                           [2, 1]])
        self.Xs = np.array([[0.5, 0.5],
                            [1.5, 0]])
        self.Y = np.random.randn(self.X.shape[0], 1)
        self.dim = self.X.shape[1]
        self.grid_dims = [[0, 1]]

        self.kernel = GPy.kern.Exponential(input_dim=self.dim, lengthscale=2,
                                           variance=2, ARD=False)
        self.kernel2 = self.kernel.copy()

        self.m = GPy.models.GPRegressionGrid(self.X, self.Y, self.kernel,
                                             grid_dims=self.grid_dims)
        self.m2 = GPy.models.GPRegression(self.X, self.Y, self.kernel2)


class GridModelRBFFactorizationTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.X = np.array([[0, 0, 0],
                           [0, 0, 1],
                           [0, 1, 0],
                           [0, 1, 1],
                           [1, 0, 0],
                           [1, 0, 1],
                           [1, 1, 0],
                           [1, 1, 1]])
        self.Xs = np.array([[0, 0, 2],
                            [-1, 3, -4]])
        self.Y = np.random.randn(8, 1)
        self.dim = self.X.shape[1]

        self.kernel = GPy.kern.RBF(input_dim=self.dim, lengthscale=(2, 3, 4),
                                   variance=2, ARD=True)
        self.kernel2 = self.kernel.copy()

        self.m = GPy.models.GPRegressionGrid(self.X, self.Y, self.kernel)
        self.m2 = GPy.models.GPRegression(self.X, self.Y, self.kernel2)

    def test_alpha_match(self):
        np.testing.assert_almost_equal(self.m.posterior.alpha,
                                       self.m2.posterior.woodbury_vector)

    def test_gradient_match(self):
        dL_dVar, dL_dLen = contract_product_gradients(self.m.kern)

        np.testing.assert_almost_equal(dL_dLen,
                                       self.m2.kern.lengthscale.gradient)
        np.testing.assert_almost_equal(self.m.likelihood.variance.gradient,
                                       self.m2.likelihood.variance.gradient)
        np.testing.assert_almost_equal(dL_dVar,
                                       self.m2.kern.variance.gradient)

    def test_likelihood(self):
        np.testing.assert_almost_equal(self.m.likelihood.variance,
                                       self.m2.likelihood.variance)
        np.testing.assert_almost_equal(self.m.log_likelihood(),
                                       self.m2.log_likelihood())

    def test_prediction_match(self):
        np.testing.assert_almost_equal(self.m.predict(self.Xs),
                                       self.m2.predict(self.Xs))


def contract_product_gradients(kern):
    D = len(kern.parts)
    dL_dLen = np.zeros((D,))
    dL_dVar_part = np.zeros((D,))
    var_parts = np.zeros((D,))
    for i, part in enumerate(kern.parts):
        dL_dLen[i] = part.lengthscale.gradient
        dL_dVar_part[i] = part.variance.gradient
        var_parts[i] = part.variance

    dVar_part_dVar_product = 1./D * var_parts ** (1 - D)
    dL_dVar = np.dot(dL_dVar_part.T, dVar_part_dVar_product)
    return dL_dVar, dL_dLen


class KernelFactorizationTest(unittest.TestCase):
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


class GridModelProdTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
#        self.X = np.array([[0, 1],
#                           [0, 2],
#                           [3, 1],
#                           [3, 2],
#                           [6, 1],
#                           [6, 2]])
        self.X = np.arange(6).reshape((3, 2))
        self.Xs = np.array([[1, 0],
                            [2, 1]])
        self.Y = np.random.randn(self.X.shape[0], 1)

        self.kernel = (GPy.kern.RBF(input_dim=1, active_dims=[0],
                                    variance=2, lengthscale=3) *
                       GPy.kern.RBF(input_dim=1, active_dims=[1],
                                    variance=2, lengthscale=3))
        self.kernel2 = self.kernel.copy()
        self.grid_dims = [[0, 1]]
        self.m = GPy.models.GPRegressionGrid(self.X, self.Y, self.kernel,
                                             grid_dims=self.grid_dims)
        self.m2 = GPy.models.GPRegression(self.X, self.Y, self.kernel2)

    def test_alpha_match(self):
        np.testing.assert_almost_equal(self.m.posterior.alpha,
                                       self.m2.posterior.woodbury_vector)

    def test_gradient_match(self):
        m_variance_gradient = [part.variance.gradient
                               for part in self.m.kern.parts]
        m_lengthscale_gradient = [part.lengthscale.gradient
                                  for part in self.m.kern.parts]

        m2_variance_gradient = [part.variance.gradient
                                for part in self.m2.kern.parts]
        m2_lengthscale_gradient = [part.lengthscale.gradient
                                   for part in self.m2.kern.parts]

        np.testing.assert_array_almost_equal(m_variance_gradient,
                                             m2_variance_gradient)
        np.testing.assert_array_almost_equal(m_lengthscale_gradient,
                                             m2_lengthscale_gradient)
        np.testing.assert_almost_equal(self.m.likelihood.variance.gradient,
                                       self.m2.likelihood.variance.gradient)

    def test_likelihood(self):
        np.testing.assert_almost_equal(self.m.likelihood.variance,
                                       self.m2.likelihood.variance)
        np.testing.assert_almost_equal(self.m.log_likelihood(),
                                       self.m2.log_likelihood())

    def test_prediction_match(self):
        np.testing.assert_almost_equal(self.m.predict(self.Xs),
                                       self.m2.predict(self.Xs))


class GridModelProdTestAdvanced(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        x = np.arange(3).reshape(3, 1)
        y = np.arange(4).reshape(2, 2)
        self.xg = [x, y]
        self.grid_dims = [[0], [1, 2]]
        self.X = np.array([[0, 0, 1],
                           [0, 2, 3],
                           [1, 0, 1],
                           [1, 2, 3],
                           [2, 0, 1],
                           [2, 2, 3]])
        self.Xs = np.array([[0.5, 0, 2],
                            [0.5, 1, 1],
                            [1.5, 0, 2],
                            [1.5, 1, 1]])
        self.Y = np.random.randn(self.X.shape[0], 1)

        self.kernel = (GPy.kern.RBF(input_dim=1, active_dims=[0],
                       variance=2, lengthscale=2) *
                       GPy.kern.RBF(input_dim=2, active_dims=[1, 2],
                       variance=3, lengthscale=[3, 4], ARD=True))
        self.kernel2 = self.kernel.copy()

        self.m = GPy.models.GPRegressionGrid(self.X, self.Y, self.kernel,
                                             grid_dims=self.grid_dims)
        self.m2 = GPy.models.GPRegression(self.X, self.Y, self.kernel2)

    def test_factor_grid(self):
        resulting_xg = factor_grid(self.X, self.grid_dims)
        np.testing.assert_(len(resulting_xg) == 2)
        np.testing.assert_array_equal(resulting_xg[0], self.xg[0])
        np.testing.assert_array_equal(resulting_xg[1], self.xg[1])

    def test_alpha_match(self):
        np.testing.assert_almost_equal(self.m.posterior.alpha,
                                       self.m2.posterior.woodbury_vector)

    def test_gradient_match(self):
        D = 2
        dL_dLen = np.zeros((D,))
        dL_dVar_part = np.zeros((D,))
        var_parts = np.zeros((D,))
        for i in [1, 2]:
            dL_dLen[i-1] = self.m.kern.parts[i].lengthscale.gradient
            dL_dVar_part[i-1] = self.m.kern.parts[i].variance.gradient
            var_parts[i-1] = self.m.kern.parts[i].variance

        dVar_part_dVar_product = 1./D * var_parts ** (1 - D)
        dL_dVar_second_factor = np.dot(dL_dVar_part.T, dVar_part_dVar_product)
        dL_dVar_second_factor = np.array(dL_dVar_second_factor).reshape((1,))

        dL_dLen_second_factor = dL_dLen

        m_variance_gradient = [self.m.kern.parts[0].variance.gradient,
                               dL_dVar_second_factor]
        m2_variance_gradient = [part.variance.gradient
                                for part in self.m2.kern.parts]
        m_lengthscale_gradient = [self.m.kern.parts[0].lengthscale.gradient,
                                  dL_dLen_second_factor]
        m2_lengthscale_gradient = [part.lengthscale.gradient
                                   for part in self.m2.kern.parts]

        np.testing.assert_array_almost_equal(m_variance_gradient,
                                             m2_variance_gradient)
        np.testing.assert_array_almost_equal(m_lengthscale_gradient[0],
                                             m2_lengthscale_gradient[0])
        np.testing.assert_array_almost_equal(m_lengthscale_gradient[1],
                                             m2_lengthscale_gradient[1])
        np.testing.assert_almost_equal(self.m.likelihood.variance.gradient,
                                       self.m2.likelihood.variance.gradient)

    def test_likelihood(self):
        np.testing.assert_almost_equal(self.m.likelihood.variance,
                                       self.m2.likelihood.variance)
        np.testing.assert_almost_equal(self.m.log_likelihood(),
                                       self.m2.log_likelihood())

    def test_prediction_match(self):
        np.testing.assert_almost_equal(self.m.predict(self.Xs),
                                       self.m2.predict(self.Xs))


class GridModelFactoredXTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        x = np.arange(3).reshape(3, 1)
        y = np.arange(4).reshape(2, 2)
        self.xg = [x, y]
        self.grid_dims = [[0], [1, 2]]
        self.X = np.array([[0, 0, 1],
                           [0, 2, 3],
                           [1, 0, 1],
                           [1, 2, 3],
                           [2, 0, 1],
                           [2, 2, 3]])
        self.xgs = [np.array([[0.5], [1.5]]),
                    np.array([[0, 2], [1, 1]])]
        self.Xs = np.array([[0.5, 0, 2],
                            [0.5, 1, 1],
                            [1.5, 0, 2],
                            [1.5, 1, 1]])
        self.Y = np.random.randn(self.X.shape[0], 1)

        self.kernel = (GPy.kern.RBF(input_dim=1, active_dims=[0],
                       variance=2, lengthscale=2) *
                       GPy.kern.Exponential(input_dim=2, active_dims=[1, 2],
                       variance=3, lengthscale=[3]))
        self.kernel2 = self.kernel.copy()

        self.m = GPy.models.GPRegressionGrid(self.xg, self.Y, self.kernel,
                                             grid_dims=self.grid_dims)
        self.m2 = GPy.models.GPRegression(self.X, self.Y, self.kernel2)

    def test_alpha_match(self):
        np.testing.assert_almost_equal(self.m.posterior.alpha,
                                       self.m2.posterior.woodbury_vector)

    def test_gradient_match(self):
        m_variance_gradient = [part.variance.gradient
                               for part in self.m.kern.parts]
        m_lengthscale_gradient = [part.lengthscale.gradient
                                  for part in self.m.kern.parts]

        m2_variance_gradient = [part.variance.gradient
                                for part in self.m2.kern.parts]
        m2_lengthscale_gradient = [part.lengthscale.gradient
                                   for part in self.m2.kern.parts]

        np.testing.assert_array_almost_equal(m_variance_gradient,
                                             m2_variance_gradient)
        np.testing.assert_array_almost_equal(m_lengthscale_gradient,
                                             m2_lengthscale_gradient)
        np.testing.assert_almost_equal(self.m.likelihood.variance.gradient,
                                       self.m2.likelihood.variance.gradient)

    def test_likelihood(self):
        np.testing.assert_almost_equal(self.m.likelihood.variance,
                                       self.m2.likelihood.variance)
        np.testing.assert_almost_equal(self.m.log_likelihood(),
                                       self.m2.log_likelihood())

    def test_prediction_match(self):
        mu, var = self.m.predict(self.xgs)
        mu2, var2 = self.m2.predict(self.Xs)
        np.testing.assert_almost_equal(self.m.predict(self.xgs),
                                       self.m2.predict(self.Xs))


class GridModelMultiProdTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.grid_dims = [[0], [1], [2, 3]]
        self.X = np.array([[0, 0, 0, 1],
                           [0, 0, 2, 3],
                           [0, 1, 0, 1],
                           [0, 1, 2, 3],
                           [0, 2, 0, 1],
                           [0, 2, 2, 3],
                           [1, 0, 0, 1],
                           [1, 0, 2, 3],
                           [1, 1, 0, 1],
                           [1, 1, 2, 3],
                           [1, 2, 0, 1],
                           [1, 2, 2, 3]])
        self.Xs = np.array([[0, 0.5, 0, 2],
                            [0, 0.5, 1, 1],
                            [1, 1.5, 0, 2],
                            [1, 1.5, 1, 1]])
        self.Y = np.random.randn(self.X.shape[0], 1)

        k1 = GPy.kern.RBF(input_dim=1, active_dims=[0],
                          variance=2, lengthscale=3)
        k2 = GPy.kern.RBF(input_dim=1, active_dims=[1],
                          variance=4, lengthscale=5)
#        k3 = (GPy.kern.Exponential(input_dim=1, active_dims=2,
#                                   variance=6, lengthscale=7) *
#              GPy.kern.Matern32(input_dim=1, active_dims=3,
#                                variance=8, lengthscale=9))
#        self.kernel = k1 * k2 * k3  # Fails due to GPy bug?
        k3 = GPy.kern.Exponential(input_dim=1, active_dims=2,
                                  variance=6, lengthscale=7)
        k4 = GPy.kern.Matern32(input_dim=1, active_dims=3,
                               variance=8, lengthscale=9)
        self.kernel = k1 * k2 * k3 * k4
        self.kernel2 = self.kernel.copy()

        self.m = GPy.models.GPRegressionGrid(self.X, self.Y, self.kernel,
                                             grid_dims=self.grid_dims)
        self.m2 = GPy.models.GPRegression(self.X, self.Y, self.kernel2)

    def test_alpha_match(self):
        np.testing.assert_almost_equal(self.m.posterior.alpha,
                                       self.m2.posterior.woodbury_vector)

    def test_gradient_match(self):
        m_variance_gradient = [part.variance.gradient
                               for part in self.m.kern.parts]
        m_lengthscale_gradient = [part.lengthscale.gradient
                                  for part in self.m.kern.parts]

        m2_variance_gradient = [part.variance.gradient
                                for part in self.m2.kern.parts]
        m2_lengthscale_gradient = [part.lengthscale.gradient
                                   for part in self.m2.kern.parts]

        np.testing.assert_array_almost_equal(m_variance_gradient,
                                             m2_variance_gradient)
        np.testing.assert_array_almost_equal(m_lengthscale_gradient,
                                             m2_lengthscale_gradient)
        np.testing.assert_almost_equal(self.m.likelihood.variance.gradient,
                                       self.m2.likelihood.variance.gradient)

    def test_likelihood(self):
        np.testing.assert_almost_equal(self.m.likelihood.variance,
                                       self.m2.likelihood.variance)
        np.testing.assert_almost_equal(self.m.log_likelihood(),
                                       self.m2.log_likelihood())

    def test_prediction_match(self):
        np.testing.assert_almost_equal(self.m.predict(self.Xs),
                                       self.m2.predict(self.Xs))
