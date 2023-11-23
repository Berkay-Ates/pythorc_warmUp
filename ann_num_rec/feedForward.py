# MNIST
# DataLoader, Transformation
# Multilayer Neural NEt, activaton function
# Loss and Optimizer
# Training Loop (batch training)
# Model evalution
# Gpu Support

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# * device configuraions
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# * hyper parameters
input_size = 784  # 28*28
hidden_size = 100
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001


# Mnist dataset
train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, transform=transforms.ToTensor(), download=False
)
test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, transform=transforms.ToTensor()
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False
)

for samples, labels in train_loader:
    print(samples.shape, labels.shape)
    break


for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(samples[i][0], cmap="gray")

plt.show()


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out


model = NeuralNet(input_size, hidden_size, num_classes).to(device)


# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training loop
n_totoal_steps = len(train_loader)
for epoc in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 100 ,1 ,28,28
        # 100, 784
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)

        # forward pass
        output = model(images)
        loss = criterion(output, labels)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(
                f"epoc {epoc+1}/ {num_epochs}, step {i+1}/{n_totoal_steps},loss = {loss.item():.4f}"
            )

# testing and evalutating
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labes in test_loader:
        images = images.reshape(-1, 28 * 28).to(device)
        labes = labels.to(device)
        output = model(images)

        _, predictions = torch.max(output, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

    acc = (100.0 * n_correct) / n_samples
    print(f"accuracy = {acc}")
