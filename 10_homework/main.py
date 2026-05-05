import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class CNN_WithBN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)

        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = self.fc2(x)
        return x

class CNN_Base(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(model, train_loader, optimizer, criterion, epochs=3):
    losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)

        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    return losses


def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total

import matplotlib.pyplot as plt

base_model = CNN_Base()
opt1 = torch.optim.Adam(base_model.parameters(), lr=0.001)

base_losses = train(base_model, train_loader, opt1, nn.CrossEntropyLoss())

bn_model = CNN_WithBN()
opt2 = torch.optim.Adam(bn_model.parameters(), lr=0.001)

bn_losses = train(bn_model, train_loader, opt2, nn.CrossEntropyLoss())

# plot
plt.plot(base_losses, label="Без нормалізації")
plt.plot(bn_losses, label="BatchNorm + Dropout")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Порівняння моделей")
plt.show()


import pandas as pd

base_acc = evaluate(base_model, test_loader)
bn_acc = evaluate(bn_model, test_loader)

base_loss = base_losses[-1]
bn_loss = bn_losses[-1]

df = pd.DataFrame({
    "Конфігурація": ["Без нормалізації", "BatchNorm + Dropout"],
    "Accuracy": [base_acc, bn_acc],
    "Loss": [base_loss, bn_loss],
    "Примітка": ["вихідна модель", "поліпшена стабільність"]
})

print(df)