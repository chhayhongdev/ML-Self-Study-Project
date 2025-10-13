#!/usr/bin/env python3
"""
Chapter 7: Best Practices and Troubleshooting - Code Examples
CNN Masterclass Curriculum

This file contains runnable examples of best practices and troubleshooting
techniques for CNN training and deployment.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import time
import json
from datetime import datetime
import random
if torch.backends.mps.is_available():
    from torch.amp import autocast, GradScaler
else:
    # Fallback for CPU or when MPS not available
    class autocast:
        def __init__(self, device_type=None):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Device configuration
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# UTILITY CLASSES AND FUNCTIONS
# ============================================================================

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True  # Stop training

        return False

    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

class SimpleCNN(nn.Module):
    """Simple CNN for demonstration"""
    def __init__(self, dropout_rate=0.5):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def get_data_loaders(batch_size=64, augmentation=True):
    """Get CIFAR-10 data loaders with optional augmentation"""
    if augmentation:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Load datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )

    # Split train into train/val
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader

def train_epoch(model, train_loader, criterion, optimizer, device, scaler=None):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        if scaler:  # Mixed precision
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# ============================================================================
# DEMONSTRATION FUNCTIONS
# ============================================================================

def demo_early_stopping():
    """Demonstrate early stopping"""
    print("\n" + "="*50)
    print("DEMO: Early Stopping")
    print("="*50)

    # Setup
    model = SimpleCNN().to(device)
    train_loader, val_loader, _ = get_data_loaders(batch_size=128, augmentation=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)

    # Training loop with early stopping
    train_losses, val_losses = [], []
    best_epoch = 0

    for epoch in range(50):  # Max epochs
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1:2d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Early stopping check
        if early_stopping(val_loss, model):
            print(f"Early stopping triggered at epoch {epoch+1}")
            best_epoch = epoch + 1
            break

    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.axvline(x=best_epoch-1, color='red', linestyle='--', label='Early Stopping')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Early Stopping Demonstration')
    plt.show()

    return model

def demo_overfitting_solutions():
    """Demonstrate solutions to overfitting"""
    print("\n" + "="*50)
    print("DEMO: Overfitting Solutions")
    print("="*50)

    # Compare models with different regularization
    configs = [
        {'dropout': 0.0, 'augmentation': False, 'weight_decay': 0.0, 'label': 'No Regularization'},
        {'dropout': 0.5, 'augmentation': False, 'weight_decay': 0.0, 'label': 'Dropout Only'},
        {'dropout': 0.5, 'augmentation': True, 'weight_decay': 1e-4, 'label': 'Full Regularization'}
    ]

    results = {}

    for config in configs:
        print(f"\nTraining with: {config['label']}")

        # Create model and data loaders
        model = SimpleCNN(dropout_rate=config['dropout']).to(device)
        train_loader, val_loader, _ = get_data_loaders(
            batch_size=128,
            augmentation=config['augmentation']
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=0.001,
            weight_decay=config['weight_decay']
        )

        # Quick training (5 epochs)
        for epoch in range(5):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)

            if epoch == 4:  # Last epoch results
                results[config['label']] = {
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                    'gap': train_acc - val_acc
                }

                print(".2f"
                      ".2f")

    # Plot results
    labels = list(results.keys())
    train_accs = [results[label]['train_acc'] for label in labels]
    val_accs = [results[label]['val_acc'] for label in labels]
    gaps = [results[label]['gap'] for label in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.bar(x - width/2, train_accs, width, label='Train Accuracy', alpha=0.8)
    ax1.bar(x + width/2, val_accs, width, label='Val Accuracy', alpha=0.8)
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Training vs Validation Accuracy')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.legend()

    ax2.bar(x, gaps, color='red', alpha=0.7)
    ax2.set_ylabel('Accuracy Gap (%)')
    ax2.set_title('Overfitting Gap (Train - Val)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right')

    plt.tight_layout()
    plt.show()

def demo_mixed_precision():
    """Demonstrate mixed precision training"""
    print("\n" + "="*50)
    print("DEMO: Mixed Precision Training")
    print("="*50)

    if not torch.backends.mps.is_available():
        print("Mixed precision requires CUDA. Skipping demo.")
        return

    # Setup
    model = SimpleCNN().to(device)
    train_loader, val_loader, _ = get_data_loaders(batch_size=128, augmentation=False)
    criterion = nn.CrossEntropyLoss()

    # Standard training
    print("Standard Precision Training:")
    model_std = SimpleCNN().to(device)
    optimizer_std = optim.Adam(model_std.parameters(), lr=0.001)

    start_time = time.time()
    for epoch in range(3):
        train_loss, train_acc = train_epoch(model_std, train_loader, criterion, optimizer_std, device)
        val_loss, val_acc = validate(model_std, val_loader, criterion, device)
        print(".2f")

    std_time = time.time() - start_time

    # Mixed precision training
    print("\nMixed Precision Training:")
    model_mixed = SimpleCNN().to(device)
    optimizer_mixed = optim.Adam(model_mixed.parameters(), lr=0.001)
    scaler = GradScaler()

    start_time = time.time()
    for epoch in range(3):
        train_loss, train_acc = train_epoch(model_mixed, train_loader, criterion, optimizer_mixed, device, scaler)
        val_loss, val_acc = validate(model_mixed, val_loader, criterion, device)
        print(".2f")

    mixed_time = time.time() - start_time

    print(".2f")
    print(".2f")
    print(".2f")

def demo_gradient_checking():
    """Demonstrate gradient checking"""
    print("\n" + "="*50)
    print("DEMO: Gradient Checking")
    print("="*50)

    # Create small model for testing
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1)
    ).to(device)

    # Create test input
    x = torch.randn(2, 10).to(device)
    target = torch.randn(2, 1).to(device)

    criterion = nn.MSELoss()

    def check_gradients(model, x, target, epsilon=1e-7):
        """Numerical gradient checking"""
        model.eval()

        # Compute analytical gradients
        output = model(x)
        loss = criterion(output, target)
        loss.backward()

        analytical_grads = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                analytical_grads[name] = param.grad.clone()

        # Compute numerical gradients
        numerical_grads = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                numerical_grads[name] = torch.zeros_like(param)

                for i in range(min(param.numel(), 10)):  # Check first 10 elements only
                    # Positive perturbation
                    param_flat = param.view(-1)
                    original_val = param_flat[i].item()
                    param_flat[i] = original_val + epsilon

                    output_pos = model(x)
                    loss_pos = criterion(output_pos, target)

                    # Negative perturbation
                    param_flat[i] = original_val - epsilon
                    output_neg = model(x)
                    loss_neg = criterion(output_neg, target)

                    # Numerical gradient
                    numerical_grads[name].view(-1)[i] = (loss_pos - loss_neg) / (2 * epsilon)

                    # Reset parameter
                    param_flat[i] = original_val

        # Compare gradients
        print("Gradient Check Results:")
        for name in analytical_grads:
            analytical = analytical_grads[name].view(-1)[:10]  # First 10 elements
            numerical = numerical_grads[name].view(-1)[:10]

            diff = torch.abs(analytical - numerical)
            max_diff = diff.max().item()
            print(".2e")

            if max_diff > 1e-4:
                print(f"  WARNING: Large gradient difference in {name}")
            else:
                print(f"  ✓ Gradients match for {name}")

    check_gradients(model, x, target)

def demo_hyperparameter_search():
    """Demonstrate hyperparameter search"""
    print("\n" + "="*50)
    print("DEMO: Hyperparameter Search")
    print("="*50)

    def quick_train(lr, batch_size, dropout):
        """Quick training with given hyperparameters"""
        model = SimpleCNN(dropout_rate=dropout).to(device)
        train_loader, val_loader, _ = get_data_loaders(batch_size=batch_size, augmentation=False)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Train for 2 epochs
        for epoch in range(2):
            train_epoch(model, train_loader, criterion, optimizer, device)

        # Evaluate
        _, val_acc = validate(model, val_loader, criterion, device)
        return val_acc

    # Grid search
    print("Grid Search:")
    param_grid = {
        'lr': [0.001, 0.01],
        'batch_size': [64, 128],
        'dropout': [0.3, 0.5]
    }

    best_acc = 0
    best_params = None

    from itertools import product
    for lr, batch_size, dropout in product(*param_grid.values()):
        acc = quick_train(lr, batch_size, dropout)
        print(".2f")

        if acc > best_acc:
            best_acc = acc
            best_params = {'lr': lr, 'batch_size': batch_size, 'dropout': dropout}

    print(f"\nBest Grid Search Result: {best_params} -> {best_acc:.2f}%")

    # Random search
    print("\nRandom Search:")
    best_acc_rand = 0
    best_params_rand = None

    for trial in range(8):  # 8 random trials
        lr = random.choice([0.001, 0.01, 0.1])
        batch_size = random.choice([64, 128])
        dropout = random.uniform(0.1, 0.7)

        acc = quick_train(lr, batch_size, dropout)
        print(".2f")

        if acc > best_acc_rand:
            best_acc_rand = acc
            best_params_rand = {'lr': lr, 'batch_size': batch_size, 'dropout': dropout}

    print(f"\nBest Random Search Result: {best_params_rand} -> {best_acc_rand:.2f}%")

def demo_model_saving_with_metadata():
    """Demonstrate model saving with metadata"""
    print("\n" + "="*50)
    print("DEMO: Model Saving with Metadata")
    print("="*50)

    # Train a quick model
    model = SimpleCNN().to(device)
    train_loader, val_loader, _ = get_data_loaders(batch_size=128, augmentation=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Quick training
    for epoch in range(3):
        train_epoch(model, train_loader, criterion, optimizer, device)

    # Evaluate
    _, test_acc = validate(model, val_loader, criterion, device)

    # Save with metadata
    def save_model_with_metadata(model, accuracy, config, save_path):
        """Save model with metadata"""
        metadata = {
            'version': '1.0.0',
            'timestamp': datetime.now().isoformat(),
            'accuracy': accuracy,
            'config': config,
            'framework': 'PyTorch',
            'pytorch_version': torch.__version__,
            'device': str(device),
            'model_architecture': str(model.__class__.__name__)
        }

        # Save metadata
        with open(save_path + '.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save model
        torch.save(model.state_dict(), save_path + '.pth')

        print(f"Model saved with metadata: {save_path}")
        print(f"Metadata: {json.dumps(metadata, indent=2)}")

    config = {
        'learning_rate': 0.001,
        'batch_size': 128,
        'epochs': 3,
        'optimizer': 'Adam',
        'loss_function': 'CrossEntropyLoss'
    }

    save_model_with_metadata(model, test_acc, config, 'demo_model')

# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================

def main():
    """Run all demonstrations"""
    print("CNN Masterclass - Chapter 7: Best Practices and Troubleshooting")
    print("="*70)

    # Run demonstrations
    demo_early_stopping()
    demo_overfitting_solutions()
    demo_mixed_precision()
    demo_gradient_checking()
    demo_hyperparameter_search()
    demo_model_saving_with_metadata()

    print("\n" + "="*70)
    print("All demonstrations completed!")
    print("Check the generated plots and console output for results.")
    print("="*70)

if __name__ == "__main__":
    main()