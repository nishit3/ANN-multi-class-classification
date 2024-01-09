import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as f
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

train_dataset = pd.read_csv('mnist_train.csv')
feature_matrix = torch.tensor(train_dataset.iloc[:, 1:].values).float()/255
label = torch.tensor(train_dataset.iloc[:, 0].values).long()

test_dataset = pd.read_csv('mnist_test.csv')
test_X = torch.tensor(test_dataset.iloc[:, 1:].values)/255
test_y = torch.tensor(test_dataset.iloc[:, 0].values)

train_dataset = TensorDataset(feature_matrix, label)
train_data_loader = DataLoader(train_dataset, batch_size=32, drop_last=True, shuffle=True)


class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputL = nn.Linear(784, 64)
        self.hidden1 = nn.Linear(64, 32)
        self.hidden2 = nn.Linear(32, 16)
        self.outputL = nn.Linear(16, 10)

    def forward(self, x):
        x = f.relu(self.inputL(x))
        x = f.relu(self.hidden1(x))
        x = f.relu(self.hidden2(x))
        return f.log_softmax(self.outputL(x), dim=1)


classifier = DigitClassifier()
optimizer = torch.optim.Adam(params=classifier.parameters(), lr=.01)
loss_function = nn.NLLLoss()
n_epochs = 100
test_accuracies = []
epochs = np.linspace(1, n_epochs, num=n_epochs)

for epoch_i in range(n_epochs):
    for X, y in train_data_loader:
        predictions = classifier(X)
        loss = loss_function(predictions, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    test_results = classifier(test_X)
    test_results = torch.argmax(test_results, dim=1)
    test_acc = torch.mean((test_results == test_y).float()).item() * 100
    print(f"Epoch {epoch_i+1}   Test Accuracy : {test_acc} %")
    test_accuracies.append(test_acc)

plt.plot(epochs, test_accuracies)
plt.xlabel('Epoch')
plt.ylabel('Accuracy %')
plt.show()
