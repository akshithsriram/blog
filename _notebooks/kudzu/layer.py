import numpy as np

class Layer:
    def __init__(self, name):
        self.name = name
        self.params = {}
        self.grads = {}

    def __call__(self, inputs):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

def xavier_init(input_dims, output_dims):
  
    return np.random.normal(
            scale=np.sqrt(2 / (input_dims + output_dims)),
            size=(input_dims, output_dims),
    )



def bias_init(output_dims):
    return np.zeros((output_dims,))

class Affine(Layer):
    def __init__(self, name, input_dims, output_dims):
        super().__init__(name)
        self.params['w'] = xavier_init(input_dims, output_dims)
        self.params['b'] = bias_init(output_dims) #np.random.randn(output_dims,)
        self.grads['w'] = np.zeros((input_dims, output_dims))
        self.grads['b'] = np.zeros(output_dims)
        

    def __call__(self, inputs):
        self.inputs = inputs
        return inputs@self.params['w'] + self.params['b']

    def backward(self, grad):
        # (m,n) @ (n,  1) = (m, 1)
        #print("gradshape", grad.shape)
        self.grads['w'] = self.inputs.T @ grad
        # (n, 1) @ (n, 1) = (1,)
        self.grads['b'] = np.sum(grad, axis=0)
        # return (n, 1) @ (1 , m) = (n, m)
        return grad@self.params['w'].T
    

class Activation(Layer):
    """
    An activation layer just applies a function
    elementwise to its inputs
    """
    def __init__(self, name, f, f_prime):
        super().__init__(name)
        self.f = f
        self.f_prime = f_prime

    def __call__(self, inputs):
        self.inputs = inputs
        return self.f(inputs)

    def backward(self, grad):
        # (N,1) * (N,1) = (N,1)
        return self.f_prime(self.inputs) * grad


sigmoid = lambda f: 1/(1+np.exp(-f))

def sigmoid_prime(f):
    h = sigmoid(f)
    return h*(1-h)

class Sigmoid(Activation):
    def __init__(self, name):
        super().__init__(name, sigmoid, sigmoid_prime)
        
def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    y = tanh(x)
    return 1 - y ** 2


class Tanh(Activation):
    def __init__(self, name):
        super().__init__(name, tanh, tanh_prime)
        
relu = lambda x: np.maximum(0, x)

def relu_prime(x):
    y = relu(x)
    dreludx = np.zeros(y.shape)
    dreludx[np.where(y > 0)] = 1
    return dreludx

class Relu(Activation):
    def __init__(self, name):
        super().__init__(name, relu, relu_prime)