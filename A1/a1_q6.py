# %%
import numpy as np
from datasets import load_mnist
import time

def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.divide(1, (1 + np.exp(-x)))

def softmax(x: np.ndarray) -> np.ndarray:
    exp = np.exp(x)
    return np.divide(exp,exp.sum(axis=1)[:, np.newaxis])

class MLP:
    def __init__(self, n_input: int, n_hidden: int, n_out: int, lr: float) -> None:
        self.n_input, self.n_hidden, self.n_out = n_input, n_hidden, n_out
        self.lr = lr

        self.b1 = np.zeros(n_hidden)
        self.W1 = np.random.normal(size=(n_input, n_hidden))
        self.b2 = np.zeros(n_out)
        self.W2 = np.random.normal(size=(n_hidden, n_out))

        self.A1 = None
        self.A2 = None
        self.X = None
        self.Y = None

    def forward(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = sigmoid(Z1)
        Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = softmax(Z2)    

        loss = np.sum(-Y * np.log(self.A2), axis=0)

        self.X = X
        self.Y = Y

        return loss

    def backward_and_update(self):
        dZ2 = self.A2 - self.Y
        dW2 = np.matmul(self.A1[:, :, np.newaxis], dZ2[:, np.newaxis, :])
        db2 = dZ2

        dZ1 = self.A1 * (1 - self.A1) * np.matmul(self.W2[np.newaxis, :, :], dZ2[:, :, np.newaxis]).squeeze(-1)
        dW1 = np.matmul(self.X[:, :, np.newaxis], dZ1[:, np.newaxis, :])
        db1 = dZ1

        self.W1 -= self.lr * dW1.mean(axis=0)
        self.W2 -= self.lr * dW2.mean(axis=0)
        self.b1 -= self.lr * db1.mean(axis=0)
        self.b2 -= self.lr * db2.mean(axis=0)

n_hidden = 300
epochs = 5
lr = 10e-3
n_out = 10

(xtrain, ytrain), (xval, yval), num_cls = load_mnist()
batch_size = 8
n_batches = np.ceil(xtrain.shape[0] / batch_size)

indices = np.arange(0, xtrain.shape[0], 1)
np.random.shuffle(indices)
xtrain = xtrain[indices]
ytrain = ytrain[indices]

n_input = xtrain.shape[1]
n_train = len(xtrain)

one_hots_train = np.zeros((n_train, n_out))
for i, yi in enumerate(ytrain):
    one_hots_train[i][yi] = 1

xtrain_batch = np.array_split(xtrain, n_batches)
one_hots_train_batch = np.array_split(one_hots_train, n_batches)

mlp = MLP(n_input, n_hidden, n_out, lr)

nbtch = len(xtrain_batch)
losses = list()
for epoch in range(epochs):
    epoch_losses = list()
    ts = time.time()
    for i, (x, y) in enumerate(zip(xtrain_batch, one_hots_train_batch)):
        if i % (n_batches // 10) == 0 and i!= 0:
            tei = time.time() - ts
            print(f'{i}/{nbtch}, time: {tei}')

        batch_losses = mlp.forward(x, y)
        epoch_losses.extend(batch_losses)
        mlp.backward_and_update()
    
    te = time.time() - ts
    avg_loss = np.mean(epoch_losses)
    print(f"Epoch: {epoch}, loss: {avg_loss}, time: {te}s")
    losses.append(avg_loss)
