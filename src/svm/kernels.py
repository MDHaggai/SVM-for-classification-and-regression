def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x1, x2, degree=3, coef0=1):
    return (np.dot(x1, x2) + coef0) ** degree

def rbf_kernel(x1, x2, gamma=None):
    if gamma is None:
        gamma = 1.0 / x1.shape[1]  # Default value
    diff = x1 - x2
    return np.exp(-gamma * np.dot(diff, diff))

def sigmoid_kernel(x1, x2, gamma=1, coef0=1):
    return np.tanh(gamma * np.dot(x1, x2) + coef0)

def kernel_function(x1, x2, kernel_type='linear', **kwargs):
    if kernel_type == 'linear':
        return linear_kernel(x1, x2)
    elif kernel_type == 'polynomial':
        return polynomial_kernel(x1, x2, **kwargs)
    elif kernel_type == 'rbf':
        return rbf_kernel(x1, x2, **kwargs)
    elif kernel_type == 'sigmoid':
        return sigmoid_kernel(x1, x2, **kwargs)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")