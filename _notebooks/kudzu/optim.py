class Optimizer:
    def step(self, model):
        raise NotImplementedError

class GD(Optimizer):
    def __init__(self, lr=0.001):
        self.lr = lr
        
    def step(self, model):
        for layer, name, param, grad in model.params_and_grads():
            layer.params[name] = param - self.lr*grad

