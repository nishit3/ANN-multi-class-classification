import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as f
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

dataset = pd.read_csv('mnist_test.csv')
train_dataset = dataset[dataset['label'] != 7]
test_dataset = dataset[dataset['label'] == 7]

feature_matrix = (torch.tensor(train_dataset.iloc[:, 1:].values)/255).float()
label = torch.tensor(train_dataset.iloc[:, 0].values).long()

test_X = (torch.tensor(test_dataset.iloc[:, 1:].values)/255).float()
test_y = torch.tensor(test_dataset.iloc[:, 0].values).long()

train_dataset = TensorDataset(feature_matrix, label)
train_data_loader = DataLoader(train_dataset, batch_size=32, drop_last=True, shuffle=True)

n_epochs = 100


class DigitClassifier(nn.Module):
    def __init__(self, depthLvl, nodeLvl):
        super().__init__()
        self.inputL = nn.Linear(784, nodeLvl)
        self.hiddenLs = []
        for i in range(depthLvl):
            self.hiddenLs.append(nn.Linear(nodeLvl, nodeLvl))
        self.outputL = nn.Linear(nodeLvl, 10)

    def forward(self, x):
        x = f.relu(self.inputL(x))
        for hiddenL in self.hiddenLs:
            x = f.relu(hiddenL(x))
        return f.log_softmax(self.outputL(x), dim=1)


classifier = DigitClassifier(depthLvl=3, nodeLvl=128)
optimizer = torch.optim.Adam(params=classifier.parameters(), lr=.001)
loss_func = nn.NLLLoss()

for epoch_i in range(n_epochs):
    for x, y in train_data_loader:
        predictions = classifier(x)
        loss = loss_func(predictions, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

test_results = classifier(test_X)
test_results = torch.argmax(test_results, dim=1)

plt.bar([1, 2, 3, 4, 5, 6, 8, 9, 0], [len(torch.where(test_results == 1)[0]), len(torch.where(test_results == 2)[0]), len(torch.where(test_results == 3)[0]), len(torch.where(test_results == 4)[0]), len(torch.where(test_results == 5)[0]), len(torch.where(test_results == 6)[0]), len(torch.where(test_results == 8)[0]), len(torch.where(test_results == 9)[0]), len(torch.where(test_results == 0)[0])], )
plt.xticks(range(10))
plt.show()
