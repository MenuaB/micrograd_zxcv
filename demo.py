from main.micrograd import MLP

mlp = MLP(3,[4,4,1])
Xs = [
    [2.0, 3.0, -1.0],
    [1.0, -3.0, 2.0],
    [-4.0, -4.0, -3.0],
    [1.0, 3.0, 5.0],
]
Ys = [1.0, -1.0, -1.0, 1.0]

epochs = 10
learning_rate = 0.1
for i in range(epochs):
    Ypred = [mlp(x=x) for x in Xs]
    loss = sum((yp - yt)**2 for yp,yt in zip(Ypred, Ys))
    print('Epoch: ', i, 'Loss value is: ', loss.value)
    mlp.zero_grad()
    loss.backward()
    for p in mlp.parameters():
        p.value += -learning_rate * p.grad