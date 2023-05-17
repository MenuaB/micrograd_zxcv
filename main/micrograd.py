import random
from .engine import Node

class Module:
    def parameters(self):
        """Returns an empty list of parameters."""
        return []
    
    def zero_grad(self):
        """Sets the gradients of all parameters to zero."""
        for p in self.parameters():
            p.grad = 0

class Neuron(Module):
    def __init__(self, nin):
        """Initializes a Neuron object with 'nin' input nodes."""
        self.w = [Node(random.uniform(-1, 1), _label=f'W{i}') for i in range(nin)]
        self.b = Node(random.uniform(-1, 1))

    def __call__(self, x):
        """
        Computes the output of the neuron given an input 'x'.

        Args:
            x: Input to the neuron.

        Returns:
            The output of the neuron after applying the activation function.
        """
        act = sum((xi * wi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out 
    
    def parameters(self):
        """Returns the list of parameters (weights and bias) of the neuron."""
        return self.w + [self.b]
    
class Layer(Module):
    def __init__(self, nin, nout):
        """
        Initializes a Layer object with 'nin' input nodes and 'nout' output nodes.

        Args:
            nin: Number of input nodes.
            nout: Number of output nodes.
        """
        self.nodes = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        """
        Computes the outputs of the layer given an input 'x'.

        Args:
            x: Input to the layer.

        Returns:
            The outputs of the layer as a list.
        """
        outs = [n(x) for n in self.nodes]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        """Returns the list of parameters (weights and biases) of all nodes in the layer."""
        return [p for node in self.nodes for p in node.parameters()]
    
class MLP(Module):
    def __init__(self, nin, nouts):
        """
        Initializes a Multi-Layer Perceptron (MLP) object.

        Args:
            nin: Number of input nodes.
            nouts: List containing sthe number of nodes in each layer of the MLP.
        """
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        """
        Computes the output of the MLP given an input 'x'.

        Args:
            x: Input to the MLP.

        Returns:
            The output of the MLP.
        """
        for l in self.layers:
            x = l(x)
        return x
    
    def parameters(self):
        """Returns the list of parameters (weights and biases) of all layers in the MLP."""
        return [p for layer in self.layers for p in layer.parameters()]
