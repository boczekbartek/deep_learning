# %%
from calendar import c
import numpy as np
from datasets import load_mnist
from sklearn.metrics import accuracy_score
from a1_q5 import MLP

n_hidden = 300
epochs = 5
lr = 10e-3
n_out = 10

(xtrain, ytrain), (xval, yval), num_cls = load_mnist()

# Shuffle the dataset
indices = np.arange(0, xtrain.shape[0], 1)
np.random.shuffle(indices)
xtrain = xtrain[indices]
ytrain = ytrain[indices]

n_input = xtrain.shape[1]
n_train = len(xtrain)
n_val = len(xval)

one_hots_train = np.zeros((n_train, n_out))
for i, yi in enumerate(ytrain):
    one_hots_train[i][yi] = 1

one_hots_val = np.zeros((n_val, n_out))
for i, yi in enumerate(yval):
    one_hots_val[i][yi] = 1

# %%
mlp = MLP(n_input, n_hidden, n_out, lr, minibatch=8)

train_losses = list()
val_losses = list()
train_losses_avg = list()
accuracies = list()
train_acc = list()
for epoch in range(15):
    epoch_train_losses = list()
    
    for i, (x, y) in enumerate(zip(xtrain, one_hots_train)):
        loss = mlp.forward(x, y)
        epoch_train_losses.append(loss)
        mlp.backward_and_update()
        train_losses.append(loss)
    epoch_val_losses = list()
    
    ypred = list()
    for i, (x, y) in enumerate(zip(xtrain, one_hots_train)):
        loss = mlp.forward(x, y)
        pred_label = np.argmax(mlp.A2)
        ypred.append(pred_label)
        epoch_val_losses.append(loss)
    accuracy_train = accuracy_score(ytrain, ypred)
    train_acc.append(accuracy_train)
    ypred = list()
    for i, (x, y) in enumerate(zip(xval, one_hots_val)):
        loss = mlp.forward(x, y)
        pred_label = np.argmax(mlp.A2)
        ypred.append(pred_label)
        epoch_val_losses.append(loss)
    accuracy = accuracy_score(yval, ypred)
    accuracies.append(accuracy)
    avg_train_loss = np.mean(epoch_train_losses)
    avg_val_loss = np.mean(epoch_val_losses)
    print(f"Epoch: {epoch}, train-loss: {avg_train_loss}, val-loss: {avg_val_loss}, train-acc: {accuracy_train}, val-acc: {accuracy}")
    train_losses_avg.append(avg_train_loss)
    val_losses.append(avg_val_loss)

# %%
import matplotlib.pyplot as plt

f, ax = plt.subplots(ncols=2, figsize=(10, 5))
ax[0].set_title('Training loss')
ax[0].plot(train_losses_avg, label='training')
ax[0].plot(val_losses, label='validation')
ax[0].set_ylabel('Loss')
ax[0].set_xlabel('Epoch')
ax[0].legend()
ax[0].set_title('Training accuracy')
ax[1].plot(train_acc, label='training')
ax[1].plot(accuracies, label='validation')
ax[1].set_ylabel('Accuracy')
ax[1].set_xlabel('Epoch')

plt.legend()
plt.savefig('Q7-1.png')
# %%

lr = 0.001
epochs = 5
data_sgd = dict()
for repeat in range(3):
    mlp = MLP(n_input, n_hidden, n_out, lr, minibatch=1)
    train_losses = list()
    val_losses = list()
    train_losses_avg = list()
    for epoch in range(epochs):
        epoch_train_losses = list()
        
        for i, (x, y) in enumerate(zip(xtrain, one_hots_train)):
            loss = mlp.forward(x, y)
            epoch_train_losses.append(loss)
            mlp.backward_and_update()
            train_losses.append(loss)
        epoch_val_losses = list()
        for i, (x, y) in enumerate(zip(xval, one_hots_val)):
            loss = mlp.forward(x, y)
            epoch_val_losses.append(loss)
                
        avg_train_loss = np.mean(epoch_train_losses)
        avg_val_loss = np.mean(epoch_val_losses)
        print(f"Epoch: {epoch}, train-loss: {avg_train_loss}, val-loss: {avg_val_loss}")
        train_losses_avg.append(avg_train_loss)
        val_losses.append(avg_val_loss)
    data_sgd[repeat] = (train_losses, val_losses, train_losses_avg)

# %%
data_lr = dict()
for this_lr in [0.001, 0.003, 0.01, 0.03]:
    mlp = MLP(n_input, n_hidden, n_out, lr=this_lr, minibatch=1)
    train_losses = list()
    val_losses = list()
    train_losses_avg = list()
    for epoch in range(epochs):
        epoch_train_losses = list()
        
        for i, (x, y) in enumerate(zip(xtrain, one_hots_train)):
            loss = mlp.forward(x, y)
            epoch_train_losses.append(loss)
            mlp.backward_and_update()
            train_losses.append(loss)
        epoch_val_losses = list()
        for i, (x, y) in enumerate(zip(xval, one_hots_val)):
            loss = mlp.forward(x, y)
            epoch_val_losses.append(loss)
                
        avg_train_loss = np.mean(epoch_train_losses)
        avg_val_loss = np.mean(epoch_val_losses)
        print(f"Epoch: {epoch}, train-loss: {avg_train_loss}, val-loss: {avg_val_loss}")
        train_losses_avg.append(avg_train_loss)
        val_losses.append(avg_val_loss)
    data_lr[this_lr] = (train_losses, val_losses, train_losses_avg, mlp)
# %%
plt.figure()
plt.title('Comparison of training loss for \ndifferent runs with SGD algorithm. Each run repeated 3 times')
data = list()

for run, (train_losses, val_losses, train_losses_avg) in data_sgd.items():
    for e, loss in enumerate(train_losses_avg):
        data.append(
            {'run': run,
            'epoch' : e,
            'train loss' : loss}
        )
        
df = pd.DataFrame(data)
sns.lineplot(data=df, x='epoch', y='train loss', hue='run', ci='std')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.savefig('Q7-2.png')
# %%


# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = []
for lr in [0.001, 0.003, 0.01, 0.03]:
    for epoch in range(5):
        data.append(
            {
                'epoch' : epoch,
                'lr' : lr,
                'loss' : data_lr[lr][0][epoch]
            }
        )
df = pd.DataFrame(data)
sns.color_palette()
sns.set_palette("Reds")
sns.lineplot(data=df, x='epoch', y='loss', hue='lr')
plt.title('Different learning rates')
plt.savefig('Q7-3.png')

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


data = list()
plt.figure()
plt.title('Comparison of epoch average traning loss\nfor different learning rates')
colors = iter('bgrc')
for lr, (train_losses, val_losses, train_losses_avg, mlp) in data_lr.items():
    print(len(train_losses_avg))
    plt.plot(train_losses_avg, '-o', label=str(lr), color=next(colors))
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('Q7-3.png')


n_hidden = 300
epochs = 10
lr = 0.003
n_out = 10

(xtrain, ytrain), (xtest, ytest), num_cls = load_mnist(final=True)

n_input = xtrain.shape[1]
n_train = len(xtrain)
n_test = len(xtest)

one_hots_train = np.zeros((n_train, n_out))
for i, yi in enumerate(ytrain):
    one_hots_train[i][yi] = 1

one_hots_test = np.zeros((n_test, n_out))
for i, yi in enumerate(yval):
    one_hots_test[i][yi] = 1

mlp = MLP(n_input, n_hidden, n_out, lr, minibatch=8)

train_losses = list()
val_losses = list()
train_losses_avg = list()
train_acc = list()
for epoch in range(epochs):
    epoch_train_losses = list()
    
    for i, (x, y) in enumerate(zip(xtrain, one_hots_train)):
        loss = mlp.forward(x, y)
        epoch_train_losses.append(loss)
        mlp.backward_and_update()
        train_losses.append(loss)
    
    
    ypred = list()
    for i, (x, y) in enumerate(zip(xtrain, one_hots_train)):
        loss = mlp.forward(x, y)
        pred_label = np.argmax(mlp.A2)
        ypred.append(pred_label)
        epoch_val_losses.append(loss)
    accuracy_train = accuracy_score(ytrain, ypred)
    train_acc.append(accuracy_train)
    
    avg_train_loss = np.mean(epoch_train_losses)
    avg_val_loss = np.mean(epoch_val_losses)
    print(f"Epoch: {epoch}, train-loss: {avg_train_loss}, val-loss: {avg_val_loss}, train-acc: {accuracy_train}")
    train_losses_avg.append(avg_train_loss)
    val_losses.append(avg_val_loss)

from sklearn.metrics import classification_report
ypred = list()
epoch_val_losses = list()
for i, (x, y) in enumerate(zip(xtest, one_hots_test)):
    loss = mlp.forward(x, y)
    pred_label = np.argmax(mlp.A2)
    epoch_val_losses.append(loss)
    ypred.append(pred_label)

accuracy_test = accuracy_score(ytest, ypred)

# %%

plt.title(f'Training progress - final, test loss = {np.mean(epoch_val_losses)}, test accuracy = {accuracy_test}')
plt.plot(train_losses_avg, '-o', color='green', label='train')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('Q7-4.png')

pd.DataFrame(classification_report(ytest, ypred, output_dict=True)).T
# %%
