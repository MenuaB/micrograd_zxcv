import math

class Node:
    def __init__(self, value, _children=set(), _op=[], grad=0, _label="NL"):
        """
        Initialize a Node object.
        
        Args:
        - value: The value of the node.
        - _children: A set of child nodes.
        - _op: A list of operations performed on the node.
        - grad: The gradient of the node.
        - _label: The label of the node.
        """
        self.value = value
        self.grad = 0
        self._backward= lambda : None
        self._label = _label
        self._children = _children
        self._op = _op

    def __repr__(self) -> str:
        """
        Return a string representation of the Node object.
        """
        return f"Node(value = {self.value})"
    
    def __add__(self, other):
        """
        Add two Node objects or a Node object and a scalar value.
        
        Args:
        - other: The other Node object or scalar value to be added.
        
        Returns:
        - An output Node object representing the sum of the operands.
        """
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.value + other.value, (self, other), '+')
        def _backward():
            self.grad += out.grad * 1.0
            other.grad += out.grad * 1.0
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        """
        Multiply two Node objects or a Node object and a scalar value.
        
        Args:
        - other: The other Node object or scalar value to be multiplied.
        
        Returns:
        - An output Node object representing the product of the operands.
        """
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.value * other.value, (self, other), '*')
        def _backward():
            self.grad += out.grad * other.value
            other.grad += out.grad * self.value
        out._backward = _backward
        return out
    
    def __pow__(self, k):
        """
        Raise the Node object to a power.
        
        Args:
        - k: The power to which the Node object is raised.
        
        Returns:
        - An output Node object representing the power of the operand.
        """
        out = Node(self.value ** k, (self, ), f'**{k}')
        def _backward():
            self.grad += out.grad * k * self.value ** (k-1)
        out._backward = _backward
        return out
    
    def __truediv__(self, other):
        """
        Divide two Node objects or a Node object and a scalar value.
        
        Args:
        - other: The other Node object or scalar value to be divided by.
        
        Returns:
        - An output Node object representing the division of the operands.
        """
        other = other if isinstance(other, Node) else Node(other)
        out = self * other ** -1
        return out 
    
    def exp(self):
        """
        Compute the exponential value of the Node object.
        
        Returns:
        - An output Node object representing the exponential value.
        """
        out = Node(math.exp(self.value), (self, ), 'exp')
        def _backward():
            self.grad += out.grad * out.value
        out._backward = _backward
        return out
    
    def tanh(self):
        """
        Compute the hyperbolic tangent value of the Node object.
        
        Returns:
        - An output Node object representing the hyperbolic tangent value.
        """
        x = self.value
        out = Node((math.exp(x*2)-1)/(math.exp(x*2)+1), (self, ), 'tanh')
        def _backward():
            self.grad += out.grad * (1-out.value**2)
        out._backward = _backward
        return out
    
    def __sub__(self, other):
        """
        Subtract two Node objects or a Node object and a scalar value.
        
        Args:
        - other: The other Node object or scalar value to be subtracted.
        
        Returns:
        - An output Node object representing the difference of the operands.
        """
        return self + -1.0 * other
    
    def __radd__(self, other):
        """
        Add a Node object and a scalar value.
        
        Args:
        - other: The scalar value to be added.
        
        Returns:
        - An output Node object representing the sum of the operands.
        """
        return self + other
    
    def __rmul__(self, other):
        """
        Multiply a Node object and a scalar value.
        
        Args:
        - other: The scalar value to be multiplied.
        
        Returns:
        - An output Node object representing the product of the operands.
        """
        return self * other
    
    def __neg__(self):
        """
        Negate the Node object.
        
        Returns:
        - An output Node object representing the negation of the operand.
        """
        return self * -1 
    
    @staticmethod
    def get_topo(node):
        """
        Perform a topological sort on the graph of nodes.
        
        Args:
        - node: The starting node for the topological sort.
        
        Returns:
        - A list of nodes in the topological order.
        """
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
        """
        Perform backpropagation to compute gradients of the nodes.
        """
        topo_sort = self.get_topo(self)
        self.grad = 1.0
        for node in topo_sort[::-1]:
            node._backward()
