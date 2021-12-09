# %%
import numpy as np
from datasets import load_mnist
import time

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def softmax(x: np.ndarray) -> np.ndarray:
    exp = np.exp(x)
    return exp / exp.sum()


class MLP:
    def __init__(self, n_input: int, n_hidden: int, n_out: int, lr: float, minibatch: int) -> None:
        self.n_input, self.n_hidden, self.n_out = n_input, n_hidden, n_out
        self.lr = lr

        self.b1 = np.zeros(n_hidden)
        self.W1 = np.random.normal(size=(n_input, n_hidden))
        self.b2 = np.zeros(n_out)
        self.W2 = np.random.normal(size=(n_hidden, n_out))
        
        self.minibatch = minibatch
        self.batch_counter = 0
        
        # minibatch gradient accumulators
        self.dW1 = 0
        self.dW2 = 0
        self.db1 = 0
        self.db2 = 0

        # cache
        self.A1 = None
        self.A2 = None
        self.X = None
        self.Y = None

    def forward(self, X: np.ndarray, Y: np.ndarray) -> float:
        Z1 = X.dot(self.W1) + self.b1
        self.A1 = sigmoid(Z1)
        Z2 = self.A1.dot(self.W2) + self.b2
        self.A2 = softmax(Z2)

        loss = np.sum(-Y * np.log(self.A2))

        self.X = X
        self.Y = Y
        return loss

    def backward_and_update(self):
        dZ2 = self.A2 - self.Y
        dW2 = self.A1[:, np.newaxis].dot(dZ2[np.newaxis, :])
        db2 = dZ2
        
        dZ1 = self.A1 * (1 - self.A1) * self.W2.dot(dZ2)
        dW1 = self.X[:, np.newaxis].dot(dZ1[np.newaxis, :])
        db1 = dZ1

        self.dW1 += dW1
        self.dW2 += dW2
        self.db1 += db1
        self.db2 += db2
        self.batch_counter += 1        
        if self.batch_counter % self.minibatch == 0:
            # Perform update
            self.W1 -= self.lr * self.dW1 / self.minibatch
            self.W2 -= self.lr * self.dW2 / self.minibatch
            self.b1 -= self.lr * self.db1 / self.minibatch
            self.b2 -= self.lr * self.db2 / self.minibatch
            
            self.zero_grad()
        
    def zero_grad(self):
        self.dW1 = 0
        self.dW2 = 0
        self.db1 = 0
        self.db2 = 0

if __name__ == '__main__':
    n_hidden = 300
    epochs = 5
    lr = 10e-3
    n_out = 10

    (xtrain, ytrain), (xval, yval), num_cls = load_mnist()

    n_input = xtrain.shape[1]
    n_train = len(xtrain)

    one_hots_train = np.zeros((n_train, n_out))
    for i, yi in enumerate(ytrain):
        one_hots_train[i][yi] = 1

    mlp = MLP(n_input, n_hidden, n_out, lr, minibatch=8)

    losses = list()
    all_losses = list()
    for epoch in range(epochs):
        epoch_losses = list()
        ts = time.time()
        for i, (x, y) in enumerate(zip(xtrain, one_hots_train)):
            loss = mlp.forward(x, y)
            if i % (xtrain.shape[0] // 10) == 0 and i != 0:
                tei = time.time() - ts
                print(f'{i}/{xtrain.shape[0]}, time: {tei}')
            epoch_losses.append(loss)
            mlp.backward_and_update()
            all_losses.append(loss)
        avg_loss = np.mean(epoch_losses)
        print(f"Epoch: {epoch}, loss: {avg_loss}, time: {time.time() - ts}")
        losses.append(avg_loss)

    import matplotlib.pyplot as plt

    plt.figure()
    plt.title('Training progress, minibatch=8}')
    plt.plot(losses, 'o-', label='train')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('Q5.png')