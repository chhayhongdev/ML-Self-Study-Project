# How to Use Datasets with CNNs

## 1. Built-in Datasets (Easy Start)

PyTorch's `torchvision` provides many common datasets:

```python
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Define transforms (preprocessing)
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize
])

# Load CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(
    root='./data',      # Where to store data
    train=True,         # Training set
    download=True,      # Download if not exists
    transform=transform # Apply transforms
)

test_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=64,      # How many samples per batch
    shuffle=True,       # Shuffle for training
    num_workers=2       # Parallel loading
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,      # No shuffle for testing
    num_workers=2
)

# Check dataset info
print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Classes: {train_dataset.classes}")
print(f"Sample shape: {train_dataset[0][0].shape}")  # Image shape
print(f"Sample label: {train_dataset[0][1]}")        # Label
```

## 2. Data Augmentation (Better Training)

```python
# Training transforms with augmentation
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),          # Random crop
    transforms.RandomHorizontalFlip(),             # Random flip
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Test transforms (no augmentation)
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
```

## 3. Custom Dataset (Your Own Data)

```python
import os
from PIL import Image
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)  # Folder names = class names
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.samples = []

        # Collect all image paths and labels
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, cls, img_name)
                self.samples.append((img_path, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

# Usage
custom_dataset = CustomDataset(
    root_dir='./my_images',  # Folder with subfolders for each class
    transform=train_transform
)
```

## 4. Training Loop with Real Data

```python
import torch.nn as nn
import torch.optim as optim

# Your CNN model (from chapter1_fundamentals_code.py)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize
model = SimpleCNN(num_classes=10)  # 10 classes for CIFAR-10
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training loop
num_epochs = 2
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:  # Print every 100 batches
            print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss/100:.4f}')
            running_loss = 0.0

print('Training finished!')
```

## 5. Evaluation

```python
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total:.2f}%')
```

## 6. Key Points

- **Transforms**: Always normalize your data (mean=0, std=1)
- **Batch Size**: 32-128 is common, depends on your GPU memory
- **Shuffle**: Always shuffle training data, never test data
- **num_workers**: Use 2-4 for faster loading (don't exceed CPU cores)
- **Dataset Structure**: For custom data, use folders named after classes

## 7. Common Datasets

- **CIFAR-10/100**: 32x32 images, 10/100 classes
- **MNIST**: Handwritten digits, 28x28 grayscale
- **ImageNet**: 1000 classes, large images (224x224)
- **Fashion-MNIST**: Clothing items, similar to MNIST

Try loading CIFAR-10 and training the SimpleCNN from the fundamentals lesson!