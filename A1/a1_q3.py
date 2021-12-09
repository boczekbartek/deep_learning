# %%
import math
from typing import List

X = [1.0, -1.0]
Y = [1, 0]
b1 = [0.0, 0.0, 0.0]
W1 = [[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]]
b2 = [0.0, 0.0]
W2 = [[1.0, 1.0], [-1.0, -1.0], [-1.0, -1.0]]

n_input=2
n_hidden = 3
n_out = 2

def sigmoid_scalar(Z: List[float]) -> List[float]:
    return [1 / (1 + math.exp(-x)) for x in Z]

def softmax_scalar(Z: List[float]) -> List[float]:
    return [math.exp(oi) / sum([math.exp(oj) for oj in Z]) for oi in Z]
    
# Forward pass
# Linear transformation with hidden layer
Z1 = [None] * n_hidden
for j in range(n_hidden):
    Z1[j] = sum(X[i] * W1[i][j] for i in range(n_input)) + b1[j]

# Activations for hidden layer
A1 = sigmoid_scalar(Z1)

# Linear transformation with output layer
Z2 = [None] * n_out
for j in range(n_out):
    Z2[j] = sum(A1[i] * W2[i][j] for i in range(n_out)) + b2[j]

# Activations for output layer
A2 = softmax_scalar(Z2)

# Calculate loss
trg_idx = Y.index(1)
yc = A2[trg_idx]
loss = -math.log(yc)

# Backward pass
# Gradient of activated output layer
dZ2 = [A2[i] - Y[i] for i in range(n_out)]

# Gradient of second layer weights
dW2 = [[None]*n_out for _ in range(n_hidden)]
for j in range(n_out):
    for i in range(n_hidden):
        dW2[i][j] = dZ2[j]*A1[i]

# Gradient of second layer bias
db2 = dZ2

# Gradient of first activated output
dZ1 = [None]*n_hidden
for j in range(n_hidden):
    dZ1[j] = A1[j] * (1 - A1[j]) * sum(dZ2[i]*W2[j][i] for i in range(n_out))
    
# Gradient of first layer weights
dW1 = [[None]*n_hidden for _ in range(n_input)]
for j in range(n_hidden):
    for i in range(n_input):
        dW1[i][j] = dZ1[j]*X[i]

# Gradient of first layer bias
db1 = dZ1

print(f'dW = {dW1}')
print(f'db = {db1}')
print(f'dV = {dW2}')
print(f'dc = {db2}')
