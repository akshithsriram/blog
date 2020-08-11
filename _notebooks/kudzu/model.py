class Model:
    def __init__(self, layers):
        self.layers = layers
        self.layerdict = {}
        for l in layers:
            self.layerdict[l.name]=l

    def __call__(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs

    def backward(self, grad):
        for layer in reversed(self.layers):
            #print(layer.name, "incoming", grad)
            grad = layer.backward(grad)
        return grad

    def params_and_grads(self):
        for layer in self.layers:
            for name, param in layer.params.items():
                grad = layer.grads[name]
                yield layer, name, param, grad