import unittest
from src.svm.kernels import linear_kernel, polynomial_kernel, rbf_kernel

class TestKernels(unittest.TestCase):

    def test_linear_kernel(self):
        x1 = [1, 2, 3]
        x2 = [4, 5, 6]
        expected = sum(a * b for a, b in zip(x1, x2))
        result = linear_kernel(x1, x2)
        self.assertEqual(result, expected)

    def test_polynomial_kernel(self):
        x1 = [1, 2, 3]
        x2 = [4, 5, 6]
        degree = 2
        gamma = 1
        coef0 = 1
        expected = (gamma * sum(a * b for a, b in zip(x1, x2)) + coef0) ** degree
        result = polynomial_kernel(x1, x2, degree, gamma, coef0)
        self.assertEqual(result, expected)

    def test_rbf_kernel(self):
        x1 = [1, 2, 3]
        x2 = [4, 5, 6]
        gamma = 0.5
        expected = exp(-gamma * sum((a - b) ** 2 for a, b in zip(x1, x2)))
        result = rbf_kernel(x1, x2, gamma)
        self.assertAlmostEqual(result, expected)

if __name__ == '__main__':
    unittest.main()