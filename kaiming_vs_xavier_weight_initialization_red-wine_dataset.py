import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import scipy.stats as stats
from sklearn.model_selection import train_test_split

data = pd.read_csv('winequality-red.csv', sep=";")
columns = data.keys().drop('quality')
data[columns] = data[columns].apply(stats.zscore)

labels = data.iloc[:, :-1].values
results = data.iloc[:, -1].values


train_X, test_X, train_y, test_y = train_test_split(labels, results, train_size=0.8, shuffle=True)

train_set = TensorDataset(torch.tensor(train_X).float(), torch.tensor(train_y))
test_set = TensorDataset(torch.tensor(test_X).float(), torch.tensor(test_y))

train_loader = DataLoader(train_set, batch_size=4, drop_last=True)
test_loader = DataLoader(test_set, batch_size=len(test_set.tensors[0]))
n_epochs = 600
epochs = np.linspace(start=1, stop=n_epochs, num=n_epochs)

kaiming_losses = np.zeros(n_epochs)
kaiming_train_acc = np.zeros(n_epochs)
kaiming_test_acc = np.zeros(n_epochs)

xavier_losses = np.zeros(n_epochs)
xavier_train_acc = np.zeros(n_epochs)
xavier_test_acc = np.zeros(n_epochs)

for i in range(10):
    kaiming_classifier = nn.Sequential(
        nn.Linear(len(columns), 32),
        nn.ReLU(),
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, data['quality'].max() + 1),
        nn.Softmax()
    )

    k_optimizer = torch.optim.Adam(kaiming_classifier.parameters(), lr=.001)
    k_loss_fun = nn.CrossEntropyLoss()

    for epoch_i in range(n_epochs):
        batch_acc = []
        batch_loss = []
        for X, y in train_loader:
            preds = kaiming_classifier(X)
            loss = k_loss_fun(preds, y)
            batch_loss.append(loss.item())
            batch_acc.append(torch.mean((torch.argmax(preds, dim=1) == y).float()).item() * 100.00)
            k_optimizer.zero_grad()
            loss.backward()
            k_optimizer.step()

        kaiming_losses[epoch_i] += (np.mean(batch_loss))/10.00
        kaiming_train_acc[epoch_i] += (np.mean(batch_acc))/10.00

        test_X, test_y = next(iter(test_loader))
        test_preds = kaiming_classifier(test_X)
        kaiming_test_acc[epoch_i] += (torch.mean((torch.argmax(test_preds, dim=1) == test_y).float()).item() * 100.00)/10.00
        print(
            f"{i+1}   Kaiming  Epoch: {epoch_i + 1}  Train Acc: {kaiming_train_acc[epoch_i]}      Test Acc: {kaiming_test_acc[epoch_i]}        Loss: {kaiming_losses[epoch_i]}")

    print('\n\n\n')

    xavier_classifier = nn.Sequential(
        nn.Linear(len(columns), 32),
        nn.ReLU(),
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, data['quality'].max() + 1),
        nn.Softmax()
    )

    for name, param in xavier_classifier.named_parameters():
        if 'weight' in name:
            nn.init.xavier_normal_(param.data)

    x_optimizer = torch.optim.Adam(xavier_classifier.parameters(), lr=.001)
    x_loss_fun = nn.CrossEntropyLoss()

    for epoch_i in range(n_epochs):
        batch_acc = []
        batch_loss = []
        for X, y in train_loader:
            preds = xavier_classifier(X)
            loss = x_loss_fun(preds, y)
            batch_loss.append(loss.item())
            batch_acc.append(torch.mean((torch.argmax(preds, dim=1) == y).float()).item() * 100.00)
            x_optimizer.zero_grad()
            loss.backward()
            x_optimizer.step()

        xavier_losses[epoch_i] += (np.mean(batch_loss))/10.00
        xavier_train_acc[epoch_i] += (np.mean(batch_acc))/10.00

        test_X, test_y = next(iter(test_loader))
        test_preds = xavier_classifier(test_X)
        xavier_test_acc[epoch_i] += (torch.mean((torch.argmax(test_preds, dim=1) == test_y).float()).item() * 100.00)/10.00
        print(f"{i+1}   Xavier  Epoch: {epoch_i + 1}  Train Acc: {xavier_train_acc[epoch_i]}      Test Acc: {xavier_test_acc[epoch_i]}        Loss: {xavier_losses[epoch_i]}")

plt.title("Loss")
plt.plot(epochs, kaiming_losses, label='Kaiming')
plt.plot(epochs, xavier_losses, label='Xavier')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.title("Train Accuracy")
plt.plot(epochs, kaiming_train_acc, label='Kaiming')
plt.plot(epochs, xavier_train_acc, label='Xavier')
plt.xlabel("Epoch")
plt.ylabel("Accuracy %")
plt.legend()
plt.show()

plt.title("Test Accuracy")
plt.plot(epochs, kaiming_test_acc, label='Kaiming')
plt.plot(epochs, xavier_test_acc, label='Xavier')
plt.ylabel("Accuracy %")
plt.xlabel("Epoch")
plt.legend()
plt.show()
