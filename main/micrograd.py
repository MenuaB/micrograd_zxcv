import random
from .engine import Node

class Module:
    def parameters(self):
        return []
    
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

class Neuron(Module):
    def __init__(self, nin):
        self.w = [Node(random.uniform(-1,1), _label=f'W{i}') for i in range(nin)]
        self.b = Node(random.uniform(-1,1))
        # print('w',len(self.w))

    def __call__(self, x):
        act = sum((xi*wi for wi,xi in zip(self.w, x)), self.b) 
        out = act.tanh()
        return out 
    
    def parameters(self):
        return self.w +[self.b]
    
class Layer(Module):
    def __init__(self, nin, nout):
        self.nodes = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.nodes]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return [p for node in self.nodes for p in node.parameters()]
    
class MLP(Module):
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for l in self.layers:
            x = l(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]