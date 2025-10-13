import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import time

print("Chapter 3: Building Your First CNN - Implementation")
print("=" * 60)

# 3.1 Device Configuration
# Device configuration
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# 3.2 Data Loading and Preprocessing
print("\n3.2 Loading CIFAR-10 Dataset")

# Data transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load datasets
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
    num_workers=0  # Changed from 2 to 0 for macOS compatibility
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0  # Changed from 2 to 0 for macOS compatibility
)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Classes: {train_dataset.classes}")
print(f"Sample shape: {train_dataset[0][0].shape}")

# 3.3 Building the CNN Architecture
print("\n3.3 Building SimpleCNN Architecture")


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # Conv layer 1: 32x32x3 -> 28x28x6 -> 14x14x6
        x = self.pool(F.relu(self.conv1(x)))

        # Conv layer 2: 14x14x6 -> 10x10x16 -> 5x5x16
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten: 5x5x16 = 400 features
        x = x.view(-1, 16 * 5 * 5)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Initialize model
model = SimpleCNN().to(device)
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# 3.4 Loss Function and Optimizer
print("\n3.4 Setting up Loss and Optimizer")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"Loss function: {criterion}")
print(f"Optimizer: {optimizer}")


# 3.5 Training Function
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

        # Print progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            batch_loss = running_loss / 100
            batch_acc = 100. * correct / total
            print(f'  Batch {batch_idx+1:3d}/{len(train_loader)}, Loss: {batch_loss:.4f}, '
                  f'Acc: {batch_acc:.2f}%')
            running_loss = 0.0

    epoch_acc = 100. * correct / total
    return epoch_acc


# 3.6 Evaluation Function
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


# 3.7 Per-Class Evaluation
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
                label = targets[i].item()
                pred = predicted[i].item()
                if label == pred:
                    class_correct[label] += 1
                class_total[label] += 1

    print("\nPer-class accuracy:")
    for i, class_name in enumerate(classes):
        if class_total[i] > 0:
            accuracy = 100 * class_correct[i] / class_total[i]
            print(f'  {class_name:8s}: {accuracy:.2f}%')


# 3.8 Training Loop
print("\n3.8 Training the Model")

num_epochs = 5  # Reduced for demo - increase to 10-20 for better results
start_time = time.time()

for epoch in range(num_epochs):
    epoch_start = time.time()

    print(f'\nEpoch {epoch+1}/{num_epochs}')
    train_accuracy = train_epoch(model, train_loader, criterion, optimizer, device)

    # Evaluate on test set
    test_accuracy, test_loss = evaluate(model, test_loader, criterion,
                                        device)

    epoch_time = time.time() - epoch_start
    print(f'  Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%, '
          f'Test Loss: {test_loss:.4f}, Time: {epoch_time:.1f}s')

total_time = time.time() - start_time
print(f'\nTraining completed in {total_time:.1f} seconds!')

# 3.9 Final Evaluation
print("\n3.9 Final Evaluation")

final_accuracy, final_loss = evaluate(model, test_loader, criterion, device)
print(f'Final Test Accuracy: {final_accuracy:.2f}%')
print(f'Final Test Loss: {final_loss:.4f}')

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
evaluate_per_class(model, test_loader, device, classes)

# 3.10 Save Model
print("\n3.10 Saving Model")

torch.save(model.state_dict(), 'simple_cnn_cifar10.pth')
print("Model saved as 'simple_cnn_cifar10.pth'")

# Demonstrate loading
print("\nDemonstrating model loading...")
loaded_model = SimpleCNN().to(device)
loaded_model.load_state_dict(torch.load('simple_cnn_cifar10.pth'))
loaded_model.eval()

# Quick test with loaded model
test_accuracy_loaded, _ = evaluate(loaded_model, test_loader, criterion, device)
print(f"Loaded model test accuracy: {test_accuracy_loaded:.2f}%")

print("\n" + "="*60)
print("Chapter 3 Summary:")
print("- Built a complete CNN from scratch")
print("- Trained on CIFAR-10 dataset")
print("- Achieved ~60-70% test accuracy (with 5 epochs)")
print("- Learned model saving/loading")
print("- Evaluated per-class performance")
print("\nNext: Chapter 4 - Advanced Techniques!")
