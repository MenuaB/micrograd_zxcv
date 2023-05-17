from main.engine import Node
import torch

x1 = Node(2.0, _label='x1')
x2 = Node(0.0, _label='x2')
w1 = Node(-3.0, _label='w1')
w2 = Node(1.0, _label='w2')
b = Node(6.8813735870195432, _label='b')

x1w1 = x1*w1; x1w1._label='x1w1'
x2w2 = x2*w2; x2w2._label='x2w2'
x1w1x2w2 = x1w1 + x2w2 ; x1w1x2w2._label='x1w1x2w2'
n = x1w1x2w2 + b; n._label = 'n'
e = (2*n).exp(); e._label = 'e'
e2 = (2*n).exp(); e2._label = 'e2'
o = (e-1)/(e+1); o._label = 'o'
o.backward()

x1t = torch.Tensor((2.0,)).type(torch.double)               ;x1t.requires_grad = True
x2t = torch.Tensor((0.0,)).type(torch.double)               ;x2t.requires_grad = True
bt = torch.Tensor((6.8813735870195432,)).type(torch.double) ;bt.requires_grad = True
w1t = torch.Tensor((-3.0,)).type(torch.double)              ;w1t.requires_grad = True
w2t = torch.Tensor((1.0,)).type(torch.double)               ;w2t.requires_grad = True
ot = (x1t*w1t)+(x2t*w2t)+bt
ot = torch.tanh(ot)
ot.backward()

tol = 1e-6
assert abs(w1.grad - w1t.grad.item()) < tol
assert abs(o.value - ot.item()) < tol