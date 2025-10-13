#!/usr/bin/env python3
"""
CNN Digit Recognition Project - Final Project
A complete CNN-based handwritten digit recognition system using MNIST dataset

This project demonstrates:
- Complete machine learning pipeline
- CNN architecture design for digit recognition
- Data preprocessing and training
- Model evaluation and deployment
- Simple web interface for digit recognition
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
import time
import json

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration (works on CPU, GPU, or Apple Silicon)
device = torch.device('mps' if torch.backends.mps.is_available() else
                     'cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_mnist_data(batch_size=64):
    """Load and preprocess MNIST dataset"""

    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize with MNIST mean/std
    ])

    # Load datasets
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

    # Split training data into train/validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader

class DigitClassifier(nn.Module):
    """Simple CNN for handwritten digit classification"""

    def __init__(self):
        super(DigitClassifier, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 28x28 -> 28x28

        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)  # 28x28 -> 14x14

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 64 channels * 7x7 spatial
        self.fc2 = nn.Linear(128, 10)  # 10 classes (digits 0-9)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # Convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 -> ReLU -> Pool

        # Flatten for fully connected layers
        x = x.view(-1, 64 * 7 * 7)

        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train the model for one epoch"""

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Print progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            print(f'Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    """Validate the model"""

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

def evaluate_model(model, test_loader, criterion, device):
    """Comprehensive model evaluation"""

    model.eval()
    all_preds = []
    all_targets = []
    test_loss = 0.0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Calculate metrics
    test_loss /= len(test_loader)
    accuracy = 100. * np.mean(np.array(all_preds) == np.array(all_targets))

    return test_loss, accuracy, all_preds, all_targets

def per_class_accuracy(predictions, targets):
    """Calculate accuracy for each digit class"""

    classes = list(range(10))
    class_correct = [0] * 10
    class_total = [0] * 10

    for pred, target in zip(predictions, targets):
        class_total[target] += 1
        if pred == target:
            class_correct[target] += 1

    print("\nPer-class accuracy:")
    for i in range(10):
        acc = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        print(".1f")

def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    """Plot training history"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Loss plot
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy plot
    ax2.plot(train_accs, label='Training Accuracy')
    ax2.plot(val_accs, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def show_predictions(model, test_loader, device, num_images=10):
    """Show model predictions on test images"""

    model.eval()
    dataiter = iter(test_loader)
    images, labels = next(dataiter)

    # Get predictions
    with torch.no_grad():
        images = images.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)

    # Plot images with predictions
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()

    for i in range(num_images):
        img = images[i].cpu().numpy().squeeze()
        img = img * 0.3081 + 0.1307  # Denormalize

        axes[i].imshow(img, cmap='gray')
        true_label = labels[i].item()
        pred_label = predicted[i].item()

        color = 'green' if true_label == pred_label else 'red'
        axes[i].set_title(f'True: {true_label}, Pred: {pred_label}', color=color)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

def save_model(model, model_path='digit_classifier.pth', metadata=None):
    """Save the trained model with metadata"""

    # Create metadata
    if metadata is None:
        metadata = {
            'model_name': 'DigitClassifier',
            'dataset': 'MNIST',
            'input_shape': '(1, 28, 28)',
            'num_classes': 10,
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }

    # Save model state and metadata
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'metadata': metadata
    }

    torch.save(checkpoint, model_path)
    print(f"Model saved to {model_path}")

    # Save metadata separately
    with open(model_path.replace('.pth', '_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

def load_model(model_path='digit_classifier.pth'):
    """Load a saved model"""

    # Load checkpoint
    checkpoint = torch.load(model_path, weights_only=False)

    # Create model and load state
    model = DigitClassifier()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Get metadata
    metadata = checkpoint.get('metadata', {})

    print(f"Model loaded from {model_path}")
    print(f"Model info: {metadata}")

    return model, metadata

def predict_digit(model, image_tensor, device):
    """Predict digit from a single image tensor"""

    model.eval()

    # Ensure image is on the right device and has batch dimension
    if image_tensor.dim() == 3:  # (C, H, W)
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    return predicted_class.item(), confidence.item(), probabilities.squeeze().cpu().numpy()

def main():
    """Main function to run the complete digit recognition project"""

    print("🎯 CNN Digit Recognition Project")
    print("=" * 50)
    print("Building a complete CNN for handwritten digit recognition")
    print()

    # Step 1: Load data
    print("Step 1: Loading MNIST dataset...")
    train_loader, val_loader, test_loader = load_mnist_data()
    print("✓ Dataset loaded successfully!")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print()

    # Step 2: Create model
    print("Step 2: Creating CNN model...")
    model = DigitClassifier().to(device)
    print("Model architecture:")
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Step 3: Setup training
    print("Step 3: Setting up training...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 3  # Reduced for demo
    print("✓ Training setup complete!")
    print()

    # Step 4: Train the model
    print("Step 4: Training the model...")
    print("-" * 30)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    start_time = time.time()

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

    total_time = time.time() - start_time
    print(".2f")
    print()

    # Step 5: Evaluate the model
    print("Step 5: Evaluating on test set...")
    test_loss, test_acc, predictions, targets = evaluate_model(model, test_loader, criterion, device)

    print(".2f")
    print(".2f")

    per_class_accuracy(predictions, targets)
    print()

    # Step 6: Visualize results
    print("Step 6: Creating visualizations...")
    try:
        plot_training_history(train_losses, val_losses, train_accs, val_accs)
        show_predictions(model, test_loader, device)
        print("✓ Visualizations created!")
    except Exception as e:
        print(f"Note: Visualizations skipped (matplotlib may not be available): {e}")
    print()

    # Step 7: Save the model
    print("Step 7: Saving the model...")
    save_model(model)
    print()

    # Step 8: Test prediction
    print("Step 8: Testing prediction function...")
    sample_images, sample_labels = next(iter(test_loader))
    test_image = sample_images[0]
    test_label = sample_labels[0]

    predicted_class, confidence, probabilities = predict_digit(model, test_image, device)

    print(f"Sample prediction:")
    print(f"True label: {test_label.item()}")
    print(f"Predicted: {predicted_class}")
    print(".2f")
    print()

    # Step 9: Final summary
    print("🎉 PROJECT COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print("What we built:")
    print("✅ Complete CNN architecture for digit recognition")
    print("✅ Full training pipeline with validation")
    print("✅ Comprehensive model evaluation")
    print("✅ Model saving and loading functionality")
    print("✅ Single image prediction capability")
    print()
    print("Key Results:")
    print(".2f")
    print(".2f")
    print(f"Training time: {total_time:.1f} seconds")
    print()
    print("Next steps:")
    print("- Try the web interface (requires Flask)")
    print("- Experiment with different architectures")
    print("- Add data augmentation")
    print("- Deploy to production")
    print()
    print("🚀 Your CNN digit recognition system is ready!")

if __name__ == '__main__':
    main()