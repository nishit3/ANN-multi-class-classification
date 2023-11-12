import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import seaborn as sbn
import matplotlib.pyplot as plt

iris = sbn.load_dataset("iris")
data = torch.tensor(iris[iris.columns[0:4]].to_numpy()).float()
labels = torch.zeros(len(data)).long()
labels[iris.species == 'versicolor'] = 1
labels[iris.species == 'virginica'] = 2


class MultiClassClassifier(nn.Module):

    def __init__(self, num_of_hidden_layers, num_of_neurons):
        super().__init__()
        self.input = nn.Linear(4, num_of_neurons)
        self.hidden_layers = []
        for i in range(num_of_hidden_layers):
            self.hidden_layers.append(nn.Linear(num_of_neurons, num_of_neurons))
        self.output = nn.Linear(num_of_neurons, 3)

    def forward(self, x):
        x = self.input(x)
        x = f.relu(x)
        for indx, hidden_layer in enumerate(self.hidden_layers):
            x = hidden_layer(x)
            x = f.relu(x)
        return self.output(x)


max_model_width = 100
max_model_depth = 5
accuracies = np.zeros((max_model_depth, max_model_width))

for depth in range(max_model_depth):
    for width in range(max_model_width):

        ANN_classifier = MultiClassClassifier(depth + 1, width + 1)
        lossFun = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(ANN_classifier.parameters(), lr=.01)

        for epoch_index in range(2500):
            label_predictions = ANN_classifier(data)
            loss = lossFun(label_predictions, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        predicted_labels = ANN_classifier(data)
        result = torch.argmax(predicted_labels, axis=1)
        result = result[result == labels]
        accuracy = len(result) / len(labels) * 100
        accuracies[depth][width] = accuracy

for i in range(max_model_depth):
    plt.plot(np.arange(1, max_model_width+1), accuracies[i], label=f"Depth = {i+1}")

plt.legend()
plt.ylabel("Accuracy %")
plt.xlabel("Width (No. of neurons)")
plt.show()
