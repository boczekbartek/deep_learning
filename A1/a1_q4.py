# %%
import math
from random import gauss
from datasets import load_synth
from typing import List

def sigmoid_scalar(Z: List[float]) -> List[float]:
    return [1 / (1 + math.exp(-x)) for x in Z]

def softmax_scalar(Z: List[float]) -> List[float]:
    return [math.exp(oi) / sum([math.exp(oj) for oj in Z]) for oi in Z]

class MLP:
    def __init__(self, n_input, n_hidden, n_out, lr) -> None:        
        self.n_input, self.n_hidden, self.n_out = n_input, n_hidden, n_out
        self.lr = lr

        self.b1 = [0.0 for _ in range(n_hidden)]
        self.W1 = [[gauss(0,1) for _ in range(n_hidden)] for _ in range(n_input)]
        self.b2 = [0.0 for _ in range(n_out)]
        self.W2 = [[gauss(0,1) for _ in range(n_out)] for _ in range(n_hidden)]

        self.A1 = None
        self.A2 = None
        self.X = None
        self.Y = None

    def forward(self, X, Y):        
        Z1 = [None] * self.n_hidden
        for j in range(self.n_hidden):
            Z1[j] = sum(X[i] * self.W1[i][j] for i in range(self.n_input)) + self.b1[j]
        
        A1 = sigmoid_scalar(Z1)

        # Linear transformation with output layer
        Z2 = [None] * self.n_out
        for j in range(self.n_out):
            Z2[j] = sum(A1[i] * self.W2[i][j] for i in range(self.n_out)) + self.b2[j]

        # Activations for output layer
        A2 = softmax_scalar(Z2)

        # Calculate loss
        trg_idx = Y.index(1)
        yc = A2[trg_idx]
        loss = -math.log(yc)

        # Cache for backward
        self.X = X
        self.Y = Y
        self.A1 = A1
        self.A2 = A2
        return loss
    
    def backward_and_update(self):
        # Backward pass
        # Gradient of activated output layer
        dZ2 = [self.A2[i] - self.Y[i] for i in range(self.n_out)]

        # Gradient of second layer weights
        dW2 = [[None]*self.n_out for _ in range(self.n_hidden)]
        for j in range(self.n_out):
            for i in range(self.n_hidden):
                dW2[i][j] = dZ2[j]*self.A1[i]

        # Gradient of second layer bias
        db2 = dZ2

        # Gradient of first activated output
        dZ1 = [None]*self.n_hidden
        for j in range(self.n_hidden):
            dZ1[j] = self.A1[j] * (1 - self.A1[j]) * sum(dZ2[i]*self.W2[j][i] for i in range(self.n_out))
            
        # Gradient of first layer weights
        dW1 = [[None]*self.n_hidden for _ in range(self.n_input)]
        for j in range(self.n_hidden):
            for i in range(self.n_input):
                dW1[i][j] = dZ1[j]*self.X[i]

        # Gradient of first layer bias
        db1 = dZ1

        for j in range(self.n_hidden):
            for i in range(self.n_input):
                self.W1[i][j] -= self.lr * dW1[i][j]
            self.b1[j] -= self.lr * db1[j]
        
        for j in range(self.n_out):
            for i in range(self.n_hidden):
                self.W2[i][j] -= self.lr * dW2[i][j]
            self.b2[j] -= self.lr * db2[j]
        

n_input=2
n_hidden = 3
n_out = 2
epochs = 5
lr = 0.001

(xtrain, ytrain), (xval, yval), num_cls = load_synth()

n_train = len(xtrain)
one_hots = [[0]* n_input for _ in range(n_train)]
for i, yi in enumerate(ytrain):
    one_hots[i][yi] = 1

mlp = MLP(n_input, n_hidden, n_out, lr)

losses = list()
for epoch in range(epochs):
    epoch_losses = list()
    for x, y in zip(xtrain, one_hots):
        loss = mlp.forward(x,y)
        epoch_losses.append(loss)
        mlp.backward_and_update()
    avg_loss = sum(epoch_losses)/len(epoch_losses)
    print(f'Epoch: {epoch}, loss: {avg_loss}')
    losses.append(avg_loss)

import matplotlib.pyplot as plt

plt.figure()
plt.title('Training progress')
plt.plot(losses, 'o-', label='train')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('Q4-1.png')