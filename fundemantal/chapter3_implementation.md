# Chapter 3: Building Your First CNN - Implementation

## 3.1 Setting Up the Environment

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
```

## 3.2 Data Loading and Preprocessing

### CIFAR-10 Dataset
CIFAR-10 consists of 60,000 32x32 color images in 10 classes:
- airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- 50,000 training images, 10,000 test images

```python
# Data transforms
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL image to tensor (C, H, W) and scale to [0,1]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1,1]
])

# Load CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# Data loaders
batch_size = 64
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2
)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Classes: {train_dataset.classes}")
```

## 3.3 Building a Simple CNN Architecture

### LeNet-5 Inspired Architecture

```python
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0)  # 3 input channels (RGB)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)

        # Pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 16 channels, 5x5 feature maps
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # Conv layer 1: 32x32x3 -> 28x28x6 -> 14x14x6
        x = self.pool(F.relu(self.conv1(x)))

        # Conv layer 2: 14x14x6 -> 10x10x16 -> 5x5x16
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten: 5x5x16 = 400
        x = x.view(-1, 16 * 5 * 5)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model
model = SimpleCNN().to(device)
print(model)
```

### Understanding the Architecture

**Input:** 32×32×3 (CIFAR-10 images)
**Conv1:** 5×5 kernel, 6 filters → 28×28×6
**Pool1:** 2×2 max pooling → 14×14×6
**Conv2:** 5×5 kernel, 16 filters → 10×10×16
**Pool2:** 2×2 max pooling → 5×5×16
**Flatten:** 400 features
**FC1:** 400→120
**FC2:** 120→84
**FC3:** 84→10 (classes)

## 3.4 Loss Function and Optimizer

```python
# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print(f"Loss function: {criterion}")
print(f"Optimizer: {optimizer}")
```

## 3.5 Training Loop

```python
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Print progress
        if (batch_idx + 1) % 100 == 0:
            print(f'Batch {batch_idx+1}/{len(train_loader)}, Loss: {running_loss/100:.4f}, Acc: {100.*correct/total:.2f}%')
            running_loss = 0.0

    return 100. * correct / total

# Training
num_epochs = 10
for epoch in range(num_epochs):
    print(f'\nEpoch {epoch+1}/{num_epochs}')
    train_accuracy = train_epoch(model, train_loader, criterion, optimizer, device)
    print(f'Training Accuracy: {train_accuracy:.2f}%')
```

## 3.6 Evaluation Function

```python
def evaluate(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    avg_loss = test_loss / len(test_loader)

    return accuracy, avg_loss

# Evaluate
test_accuracy, test_loss = evaluate(model, test_loader, criterion, device)
print(f'\nTest Results:')
print(f'Accuracy: {test_accuracy:.2f}%')
print(f'Loss: {test_loss:.4f}')
```

## 3.7 Per-Class Accuracy

```python
def evaluate_per_class(model, test_loader, device, classes):
    model.eval()
    class_correct = [0] * len(classes)
    class_total = [0] * len(classes)

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            for i in range(len(targets)):
                label = targets[i]
                pred = predicted[i]
                if label == pred:
                    class_correct[label] += 1
                class_total[label] += 1

    # Print per-class accuracy
    for i, class_name in enumerate(classes):
        if class_total[i] > 0:
            accuracy = 100 * class_correct[i] / class_total[i]
            print(f'{class_name}: {accuracy:.2f}%')

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
evaluate_per_class(model, test_loader, device, classes)
```

## 3.8 Saving and Loading Models

```python
# Save model
torch.save(model.state_dict(), 'simple_cnn_cifar10.pth')
print("Model saved!")

# Load model
model = SimpleCNN().to(device)
model.load_state_dict(torch.load('simple_cnn_cifar10.pth'))
model.eval()
print("Model loaded!")
```

## 3.9 Complete Training Script

```python
# Put it all together
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

# Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleCNN().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
num_epochs = 5  # Reduced for demo
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print(f'Epoch {epoch+1}: Test Accuracy = {100.*correct/total:.2f}%')

print("Training completed!")
```

---

**Next:** Chapter 4 - Advanced CNN Techniques (Batch Normalization, Data Augmentation, ResNet)!