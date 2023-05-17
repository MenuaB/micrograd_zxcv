import math

class Node:
    def __init__(self, value, _children=set(), _op=[], grad=0, _label="NL"):
        self.value = value
        self.grad = 0
        self._backward= lambda : None
        self._label = _label
        self._children = _children
        self._op = _op

    def __repr__(self) -> str:
        return f"Node(value = {self.value})"
    
    def __add__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.value + other.value, (self, other), '+')
        def _backward():
            self.grad += out.grad * 1.0
            other.grad += out.grad * 1.0
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.value * other.value, (self, other), '*')
        def _backward():
            self.grad += out.grad * other.value
            other.grad += out.grad * self.value
        out._backward = _backward
        return out
    
    def __pow__(self, k):
        out = Node(self.value ** k, (self, ), f'**{k}')
        def _backward():
            self.grad += out.grad * k * self.value ** (k-1)
        out._backward = _backward
        return out
    
    def __truediv__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = self * other ** -1
        return out 
    
    def exp(self):
        out = Node(math.exp(self.value), (self, ), 'exp')
        def _backward():
            self.grad += out.grad * out.value
        out._backward = _backward
        return out
    
    def tanh(self):
        x = self.value
        out = Node((math.exp(x*2)-1)/(math.exp(x*2)+1), (self, ), 'tanh')
        def _backward():
            self.grad += out.grad * (1-out.value**2)
        out._backward = _backward
        return out
    
    def __sub__(self, other):
        return self + -1.0 * other
    
    def __radd__(self, other):
        return self + other
    
    def __rmul__(self, other):
        return self * other
    
    def __neg__(self):
        return self * -1 
    
    @staticmethod
    def get_topo(node):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
            return topo
        return build_topo(node)
        
    def backward(self):
        topo_sort = self.get_topo(self)
        self.grad = 1.0
        for node in topo_sort[::-1]:
            node._backward()