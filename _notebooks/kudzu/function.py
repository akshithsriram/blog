import numpy as np

class Function:

    def __init__(self):
        self.params = {}
        self.grads = {}

    def __call__(self, inputs):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def params_and_grads(self):
        for name, param in self.params.items():
            grad = self.grads[name]
            yield name, param, grad

class Affine(Function):
    def __init__(self, input_dims):
        super().__init__()
        self.params['w'] = np.random.randn(input_dims, 1)
        self.params['b'] = np.random.randn()
        self.grads['w'] = np.zeros((input_dims, 1))
        self.grads['b'] = 0.

    def __call__(self, inputs):
        self.inputs = inputs
        # (n,m) @ (m, 1)  = (n, 1)
        return inputs @ self.params['w'] + self.params['b']

    def backward(self, grad):
        # (m,n) @ (n,  1) = (m, 1)
        #print("gradshape", grad.shape)
        self.grads['w'] = self.inputs.T @ grad
        # (n, 1) @ (n, 1) = scalar
        self.grads['b'] = np.sum(grad, axis=0)
        # return (n, 1) @ (1 , m) = (n, m)
        return grad@self.params['w'].T

