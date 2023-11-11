import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

total_elements_per_class = 100

starting_xandy_coordinates_class1 = [1, 1]
starting_xandy_coordinates_class2 = [5, 1]
starting_xandy_coordinates_class3 = [1, 5]

class1 = [starting_xandy_coordinates_class1[0] + np.random.randn(total_elements_per_class), starting_xandy_coordinates_class1[1] + np.random.randn(total_elements_per_class)]
class2 = [starting_xandy_coordinates_class2[0] + np.random.randn(total_elements_per_class), starting_xandy_coordinates_class2[1] + np.random.randn(total_elements_per_class)]
class3 = [starting_xandy_coordinates_class3[0] + np.random.randn(total_elements_per_class), starting_xandy_coordinates_class3[1] + np.random.randn(total_elements_per_class)]
data_np = np.hstack((class1, class2, class3)).T

labels_np = np.concatenate((np.zeros((total_elements_per_class, 1)), np.ones((total_elements_per_class, 1)), np.ones((total_elements_per_class, 1))+1))
labels_np = labels_np.reshape(-1)
data = torch.tensor(data_np).float()
labels = torch.tensor(labels_np).long()
n_experiments = 50
accuracies_with_softmax = np.zeros(n_experiments)
accuracies_without_softmax = np.zeros(n_experiments)

for experiment in range(n_experiments):
    ANN_classifier = nn.Sequential(
        nn.Linear(2, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 3),
        nn.Softmax(dim=1)
    )

    lossFun = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(ANN_classifier.parameters(), 0.01)
    epochs = 2000

    for epoch in range(epochs):
        class_predictions = ANN_classifier(data)
        loss = lossFun(class_predictions, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    outputs_for_data = ANN_classifier(data)
    result = torch.argmax(outputs_for_data, 1)
    result = result[result == labels]
    accuracy = len(result) / len(labels) * 100
    accuracies_with_softmax[experiment] = accuracy

for experiment in range(n_experiments):
    ANN_classifier = nn.Sequential(
        nn.Linear(2, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 3),
        # nn.Softmax(dim=1)
    )

    lossFun = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(ANN_classifier.parameters(), 0.01)
    epochs = 2000

    for epoch in range(epochs):
        class_predictions = ANN_classifier(data)
        loss = lossFun(class_predictions, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    outputs_for_data = ANN_classifier(data)
    result = torch.argmax(outputs_for_data, 1)
    result = result[result == labels]
    accuracy = len(result) / len(labels) * 100
    accuracies_without_softmax[experiment] = accuracy

plt.plot(np.linspace(1, n_experiments, n_experiments), accuracies_with_softmax)
plt.ylabel('With Softmax')
plt.show()

plt.plot(np.linspace(1, n_experiments, n_experiments), accuracies_without_softmax)
plt.ylabel('Without Softmax')
plt.show()
