import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import time

print("Chapter 4: Advanced CNN Techniques")
print("=" * 60)

# Device configuration
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")


# 4.1 Batch Normalization
print("\n4.1 Batch Normalization")


class CNNWithBatchNorm(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNWithBatchNorm, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 8 * 8)
        x = self.dropout(F.relu(self.bn3(self.fc1(x))))
        x = self.fc2(x)
        return x


# 4.2 Data Augmentation
print("\n4.2 Data Augmentation")

# Training transforms with augmentation
train_transforms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Test transforms (no augmentation)
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Load data
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True,
                                             transform=train_transforms)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True,
                                            transform=test_transforms)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

# 4.3 Residual Networks (ResNet)
print("\n4.3 Residual Networks (ResNet)")


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # Residual connection
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


# 4.4 Compare Models
print("\n4.4 Comparing Different Architectures")

models = {
    'Simple CNN': CNNWithBatchNorm(),
    'ResNet18': ResNet18()
}

for name, model in models.items():
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{name}: {total_params:,} parameters")

# 4.5 Training with Advanced Techniques
print("\n4.5 Training with Advanced Techniques")


def train_epoch(model, train_loader, criterion, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    # Step the scheduler
    scheduler.step()

    return running_loss / len(train_loader), 100. * correct / total


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

    return test_loss / len(test_loader), 100. * correct / total


# Training configurations
configs = {
    'CNN + BatchNorm': {
        'model': CNNWithBatchNorm(),
        'optimizer': lambda model: optim.Adam(model.parameters(), lr=0.001,
                                              weight_decay=1e-4),
        'scheduler': lambda optimizer: torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                                   T_max=10)
    },
    'ResNet18': {
        'model': ResNet18(),
        'optimizer': lambda model: optim.SGD(model.parameters(), lr=0.01, momentum=0.9,
                                        weight_decay=5e-4),
        'scheduler': lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, step_size=5,
                                                                         gamma=0.1)
    }
}

results = {}

for name, config in configs.items():
    print(f"\nTraining {name}...")
    model = config['model'].to(device)
    optimizer = config['optimizer'](model)
    scheduler = config['scheduler'](optimizer)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    train_accs = []
    test_accs = []

    start_time = time.time()

    for epoch in range(3):  # Short training for demo
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%, LR={current_lr:.6f}")

    training_time = time.time() - start_time
    results[name] = {
        'final_test_acc': test_accs[-1],
        'training_time': training_time,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_accs': test_accs
    }

# 4.6 Results Comparison
print("\n4.6 Results Comparison")
print("-" * 40)

for name, result in results.items():
    print(f"{name}:")
    print(".2f")
    print(".1f")
    print()

# 4.7 Learning Rate Scheduling Demo
print("\n4.7 Learning Rate Scheduling Demo")

# Different schedulers
schedulers_demo = {
    'StepLR': torch.optim.lr_scheduler.StepLR(optim.SGD([torch.tensor(1.0, requires_grad=True)], lr=0.1), step_size=5, gamma=0.5),
    'ExponentialLR': torch.optim.lr_scheduler.ExponentialLR(optim.SGD([torch.tensor(1.0, requires_grad=True)], lr=0.1), gamma=0.9),
    'CosineAnnealing': torch.optim.lr_scheduler.CosineAnnealingLR(optim.SGD([torch.tensor(1.0, requires_grad=True)], lr=0.1), T_max=10)
}

plt.figure(figsize=(12, 4))

for i, (name, scheduler) in enumerate(schedulers_demo.items(), 1):
    lrs = []
    for epoch in range(20):
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()

    plt.subplot(1, 3, i)
    plt.plot(lrs)
    plt.title(f'{name} Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')

plt.tight_layout()
plt.savefig('lr_schedules.png', dpi=150, bbox_inches='tight')
print("Learning rate schedules saved as 'lr_schedules.png'")

print("\n" + "="*60)
print("Chapter 4 Summary:")
print("- Batch Normalization improves training stability")
print("- Data augmentation prevents overfitting")
print("- ResNet solves vanishing gradient problem")
print("- Learning rate scheduling optimizes convergence")
print("- Regularization techniques improve generalization")
print("\nNext: Chapter 5 - Model Evaluation and Testing!")