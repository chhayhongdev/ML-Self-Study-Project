# Convolutional Neural Networks (CNNs) Masterclass: From Theory to Deployment

## Lesson 1: Introduction to CNNs - Theory and Basics

### What are Convolutional Neural Networks?
Convolutional Neural Networks (CNNs) are a class of deep neural networks specifically designed for processing structured grid data, particularly images. Unlike traditional neural networks, CNNs exploit the spatial structure of data through convolutional layers.

### Why CNNs for Computer Vision?
- **Parameter Efficiency**: CNNs share weights across spatial locations, reducing parameters compared to fully connected networks.
- **Translation Invariance**: Detect features regardless of position in the image.
- **Hierarchical Feature Learning**: Learn low-level features (edges) to high-level features (objects).

### Key Components of a CNN
1. **Convolutional Layers**: Extract features using filters/kernels
2. **Pooling Layers**: Reduce spatial dimensions and provide translation invariance
3. **Activation Functions**: Introduce non-linearity (ReLU, Sigmoid, Tanh)
4. **Fully Connected Layers**: Classification based on extracted features
5. **Batch Normalization**: Stabilize and accelerate training
6. **Dropout**: Prevent overfitting

### CIFAR-10 Dataset
- 60,000 32x32 color images
- 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- 50,000 training images, 10,000 test images

## 🆕 **NEW: MNIST Dataset for Digit Recognition**
- **60,000 training images, 10,000 test images**
- **28×28 grayscale images** (perfect for beginners)
- **10 classes: digits 0-9**
- **Why MNIST?** Simple, well-understood, fast training
- **Real-world application**: Handwritten digit recognition (checks, forms, etc.)

## 🆕 **NEW: MNIST Dataset for Digit Recognition**
- **60,000 training images, 10,000 test images**
- **28×28 grayscale images** (perfect for beginners)
- **10 classes: digits 0-9**
- **Why MNIST?** Simple, well-understood, fast training
- **Real-world application**: Handwritten digit recognition (checks, forms, etc.)

## Lesson 2: The Mathematics Behind CNNs

### Convolution Operation
The fundamental operation in CNNs is convolution:

**Mathematical Definition:**
```
(f * g)(x,y) = ΣᵢΣⱼ f(i,j) * g(x-i, y-j)
```

Where:
- `f` is the input image/feature map
- `g` is the kernel/filter
- `*` denotes convolution

**Example with 3x3 kernel on 5x5 input:**
```
Input:     Kernel:    Output:
1 2 3 4 5   1 0 1     12 16 20
2 3 4 5 6   0 1 0  →  16 20 24
3 4 5 6 7   1 0 1     20 24 28
4 5 6 7 8
5 6 7 8 9
```

### Pooling Operations
**Max Pooling:**
```
Input:     Max Pool (2x2):
1 2 3 4    3 4
2 3 4 5 →  4 5
3 4 5 6
4 5 6 7
```

**Average Pooling:**
```
Input:     Avg Pool (2x2):
1 2 3 4    2.0 3.0
2 3 4 5 →  3.0 4.0
3 4 5 6
4 5 6 7
```

### Activation Functions
**ReLU (Rectified Linear Unit):**
```
f(x) = max(0, x)
```
- Advantages: Prevents vanishing gradient, computationally efficient
- Used in most modern CNNs

**Sigmoid:**
```
f(x) = 1 / (1 + e^(-x))
```
- Output: (0,1)
- Used in binary classification

**Softmax:**
```
f(x_i) = e^(x_i) / Σ_j e^(x_j)
```
- Converts logits to probabilities
- Used in multi-class classification

### Backpropagation in CNNs
CNNs use gradient descent with backpropagation:

**Loss Function (Cross-Entropy):**
```
L = -Σᵢ y_i * log(ŷ_i)
```

**Gradient Computation:**
- Compute ∂L/∂W for each layer
- Update weights: W = W - α * ∂L/∂W
- Chain rule applies through convolutional layers

## 🆕 **NEW: Understanding Model Parameters**
### Parameter Calculation in CNNs
For a convolutional layer:
```
Parameters = (kernel_size² × input_channels × output_channels) + output_channels
```

**Example: Conv2d(3, 32, kernel_size=3)**
```
Parameters = (3×3 × 3 × 32) + 32 = (27 × 32) + 32 = 864 + 32 = 896 parameters
```

### Memory Requirements
- **Forward pass**: Store activations for backpropagation
- **Backward pass**: Store gradients for each parameter
- **GPU Memory**: Typically 2-3x model size during training

## 🆕 **NEW: Understanding Model Parameters**
### Parameter Calculation in CNNs
For a convolutional layer:
```
Parameters = (kernel_size² × input_channels × output_channels) + output_channels
```

**Example: Conv2d(3, 32, kernel_size=3)**
```
Parameters = (3×3 × 3 × 32) + 32 = (27 × 32) + 32 = 864 + 32 = 896 parameters
```

### Memory Requirements
- **Forward pass**: Store activations for backpropagation
- **Backward pass**: Store gradients for each parameter
- **GPU Memory**: Typically 2-3x model size during training

## Lesson 3: Building Your First CNN - Implementation

### Setting Up the Environment
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### 🆕 **NEW: MNIST Data Loading (Beginner-Friendly)**
```python
# MNIST transforms (simpler than CIFAR-10)
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert to tensor (0-1 range)
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean/std
])

# Load MNIST dataset
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

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
```

### 🆕 **NEW: Simple CNN for MNIST**
```python
class DigitClassifier(nn.Module):
    """Simple CNN for handwritten digit classification"""

    def __init__(self):
        super(DigitClassifier, self).__init__()

        # Convolutional layers (1 input channel for grayscale)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 28x28 → 28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 28x28 → 28x28

        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)  # 28x28 → 14x14

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 64 channels × 7×7 spatial
        self.fc2 = nn.Linear(128, 10)  # 10 classes (digits 0-9)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # Convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 → ReLU → Pool
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 → ReLU → Pool

        # Flatten for fully connected layers
        x = x.view(-1, 64 * 7 * 7)

        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Create model
model = DigitClassifier().to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### Data Loading and Preprocessing
```python
# CIFAR-10 transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
```

### Building a Simple CNN
```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 6, 5)  # 3 input channels, 6 output, 5x5 kernel
        self.pool = nn.MaxPool2d(2, 2)   # 2x2 max pooling
        self.conv2 = nn.Conv2d(6, 16, 5)

        # Fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Conv layer 1
        x = self.pool(F.relu(self.conv1(x)))
        # Conv layer 2
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten
        x = x.view(-1, 16 * 5 * 5)
        # FC layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleCNN().to(device)
```

### Training Loop
```python
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Training function
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(loader), 100. * correct / total

# Training
num_epochs = 10
for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    print(f'Epoch {epoch+1}/{num_epochs}: Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
```

## 🆕 **NEW: Complete MNIST Training Example**
```python
def main():
    """Complete MNIST training pipeline"""
    print("🎯 MNIST Digit Recognition Training")
    print("=" * 40)

    # 1. Load data
    print("Loading MNIST dataset...")
    train_loader, test_loader = load_mnist_data()
    print("✓ Data loaded successfully!")

    # 2. Create model
    print("Creating CNN model...")
    model = DigitClassifier().to(device)
    print(f"Model: {model}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 3. Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 5

    # 4. Training loop
    print("Training model...")
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)

        # Validation
        val_loss, val_acc = validate(model, test_loader, criterion)

        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(".4f")
        print(".4f")

    # 5. Final evaluation
    test_loss, test_acc, predictions, targets = evaluate_model(model, test_loader, criterion)
    print(".4f")

    # 6. Save model
    save_model(model, 'digit_classifier.pth')
    print("✓ Model saved!")

    return model

# Run training
if __name__ == '__main__':
    trained_model = main()
```

## 🆕 **NEW: Complete MNIST Training Example**
```python
def main():
    """Complete MNIST digit recognition training pipeline"""
    print("🎯 MNIST Digit Recognition Training")
    print("=" * 40)

    # 1. Load data
    print("Loading MNIST dataset...")
    train_loader, test_loader = load_mnist_data()
    print("✓ Data loaded successfully!")

    # 2. Create model
    print("Creating CNN model...")
    model = DigitClassifier().to(device)
    print(f"Model: {model}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 3. Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 5

    # 4. Training loop
    print("Training model...")
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)

        # Validation
        val_loss, val_acc = validate(model, test_loader, criterion)

        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(".4f")
        print(".4f")

    # 5. Final evaluation
    test_loss, test_acc, predictions, targets = evaluate_model(model, test_loader, criterion)
    print(".4f")

    # 6. Save model
    save_model(model, 'digit_classifier.pth')
    print("✓ Model saved!")

    return model

# Run training
if __name__ == '__main__':
    trained_model = main()
```

## Lesson 4: Advanced CNN Techniques

### Batch Normalization Implementation
```python
class CNNWithBN(nn.Module):
    def __init__(self):
        super(CNNWithBN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 8 * 8)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
```

### Data Augmentation
```python
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
```

### Residual Networks (ResNet)
```python
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
```

## 🆕 **NEW: Understanding Overfitting and Underfitting**
### Signs of Overfitting:
- Training accuracy high (>95%), validation accuracy low/stagnant
- Large gap between train and validation loss
- Model performs well on training data but poorly on new data

### Signs of Underfitting:
- Both training and validation accuracy low
- Model unable to learn basic patterns
- High bias, low variance

### Solutions:
**For Overfitting:**
- Add dropout layers
- Use data augmentation
- Implement early stopping
- Reduce model complexity
- Add regularization (L1/L2)

**For Underfitting:**
- Increase model capacity (more layers/filters)
- Train longer (more epochs)
- Reduce regularization
- Use better optimization (Adam vs SGD)
- Adjust learning rate

## 🆕 **NEW: Understanding Overfitting and Underfitting**
### Signs of Overfitting:
- Training accuracy high (>95%), validation accuracy low/stagnant
- Large gap between train and validation loss
- Model performs well on training data but poorly on new data

### Signs of Underfitting:
- Both training and validation accuracy low
- Model unable to learn basic patterns
- High bias, low variance

### Solutions:
**For Overfitting:**
- Add dropout layers
- Use data augmentation
- Implement early stopping
- Reduce model complexity
- Add regularization (L1/L2)

**For Underfitting:**
- Increase model capacity (more layers/filters)
- Train longer (more epochs)
- Reduce regularization
- Use better optimization (Adam vs SGD)
- Adjust learning rate

## Lesson 5: Model Evaluation and Testing

### Evaluation Metrics
```python
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Per-class accuracy
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1

    # Overall accuracy
    accuracy = 100. * correct / total
    print(f'Overall Accuracy: {accuracy:.2f}%')

    # Per-class accuracy
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    for i in range(10):
        if class_total[i] > 0:
            print(f'{classes[i]}: {100. * class_correct[i] / class_total[i]:.2f}%')

    return accuracy

# Evaluate the model
test_accuracy = evaluate(model, test_loader)
```

### Confusion Matrix
```python
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(model, test_loader, classes):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
plot_confusion_matrix(model, test_loader, classes)
```

## 🆕 **NEW: Precision, Recall, and F1-Score**
```python
from sklearn.metrics import classification_report, precision_recall_fscore_support

def detailed_metrics(model, test_loader, classes):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Detailed classification report
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=classes))

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)

    print("\\nPer-class Metrics:")
    for i, class_name in enumerate(classes):
        print("10s")

# For MNIST digits
mnist_classes = [str(i) for i in range(10)]
detailed_metrics(model, test_loader, mnist_classes)
```

## 🆕 **NEW: Precision, Recall, and F1-Score**
```python
from sklearn.metrics import classification_report, precision_recall_fscore_support

def detailed_metrics(model, test_loader, classes):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Detailed classification report
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=classes))

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)

    print("\\nPer-class Metrics:")
    for i, class_name in enumerate(classes):
        print("10s")

# For MNIST digits
mnist_classes = [str(i) for i in range(10)]
detailed_metrics(model, test_loader, mnist_classes)
```

## Lesson 6: Model Deployment and Inference

### Saving and Loading Models
```python
# Save model
torch.save(model.state_dict(), 'cnn_model.pth')

# Load model
model = SimpleCNN()
model.load_state_dict(torch.load('cnn_model.pth'))
model.to(device)
model.eval()
```

### Inference on Single Images
```python
from PIL import Image

def predict_image(model, image_path, transform, classes):
    # Load and preprocess image
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)

    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = probabilities.max(1)

    predicted_class = classes[predicted.item()]
    confidence_score = confidence.item()

    return predicted_class, confidence_score

# Example usage
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
predicted_class, confidence = predict_image(model, 'test_image.jpg', transform, classes)
print(f'Predicted: {predicted_class} (Confidence: {confidence:.2f})')
```

### 🆕 **NEW: Web Deployment with FastAPI**
```python
# app.py - FastAPI deployment
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
import base64

app = FastAPI(title="Digit Recognition API")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Device and model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = DigitClassifier()
model.load_state_dict(torch.load('digit_classifier.pth', map_location=device))
model.to(device)
model.eval()

def preprocess_image(image_data: bytes) -> torch.Tensor:
    """Preprocess image for model prediction"""
    image = Image.open(io.BytesIO(image_data)).convert('L')
    image = Image.eval(image, lambda x: 255 - x)  # Invert colors
    image = image.resize((28, 28))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    return transform(image)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict digit from uploaded image"""
    image_data = await file.read()
    image_tensor = preprocess_image(image_data)

    with torch.no_grad():
        outputs = model(image_tensor.unsqueeze(0).to(device))
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = probabilities.max(1)

    return {
        "prediction": int(predicted.item()),
        "confidence": float(confidence.item()),
        "probabilities": probabilities.squeeze().tolist()
    }

@app.get("/")
async def root():
    return {"message": "Digit Recognition API", "status": "running"}
```

### Model Optimization for Deployment
```python
# Convert to TorchScript for production
model.eval()
example_input = torch.randn(1, 3, 32, 32).to(device)
traced_model = torch.jit.trace(model, example_input)
traced_model.save('cnn_model_traced.pt')

# Load traced model
loaded_model = torch.jit.load('cnn_model_traced.pt')
```

### Performance Monitoring
```python
import time

def benchmark_inference(model, test_loader, num_runs=100):
    model.eval()
    times = []

    with torch.no_grad():
        for _ in range(num_runs):
            images, _ = next(iter(test_loader))
            images = images.to(device)

            start_time = time.time()
            _ = model(images)
            end_time = time.time()

            times.append(end_time - start_time)

    avg_time = sum(times) / len(times)
    fps = 1.0 / avg_time
    print(f'Average inference time: {avg_time:.4f}s')
    print(f'FPS: {fps:.2f}')

benchmark_inference(model, test_loader)
```

## 🆕 **NEW: Web Deployment with FastAPI**
```python
# app.py - FastAPI deployment
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
import base64

app = FastAPI(title="Digit Recognition API")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
),
)

# Device and model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = DigitClassifier()
model.load_state_dict(torch.load('digit_classifier.pth', map_location=device))
model.to(device)
model.eval()

def preprocess_image(image_data: bytes) -> torch.Tensor:
    """Preprocess image for model prediction"""
    image = Image.open(io.BytesIO(image_data)).convert('L')
    image = Image.eval(image, lambda x: 255 - x)  # Invert colors
    image = image.resize((28, 28))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    return transform(image)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict digit from uploaded image"""
    image_data = await file.read()
    image_tensor = preprocess_image(image_data)

    with torch.no_grad():
        outputs = model(image_tensor.unsqueeze(0).to(device))
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = probabilities.max(1)

    return {
        "prediction": int(predicted.item()),
        "confidence": float(confidence.item()),
        "probabilities": probabilities.squeeze().tolist()
    }

@app.get("/")
async def root():
    return {"message": "Digit Recognition API", "status": "running"}
```

## 🆕 **NEW: Docker Deployment**
```dockerfile
# Dockerfile
FROM python:3.11-slim

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy model and code
COPY digit_classifier.pth .
COPY app.py .

# Expose port
EXPOSE 8000

# Run FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  digit-recognition:
    build: .
    ports:
      - "8000:8000"
    restart: unless-stopped
```

## 🆕 **NEW: Docker Deployment**
```dockerfile
# Dockerfile
FROM python:3.11-slim

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy model and code
COPY digit_classifier.pth .
COPY app.py .

# Expose port
EXPOSE 8000

# Run FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  digit-recognition:
    build: .
    ports:
      - "8000:8000"
    restart: unless-stopped
```

## Lesson 7: Best Practices and Troubleshooting

### Training Tips
1. **Learning Rate Scheduling**: Use `torch.optim.lr_scheduler` for better convergence
2. **Early Stopping**: Monitor validation loss to prevent overfitting
3. **Regularization**: Use dropout, weight decay, and data augmentation
4. **Gradient Clipping**: Prevent exploding gradients
5. **Mixed Precision Training**: Use `torch.cuda.amp` for faster training

### Common Issues and Solutions
1. **Overfitting**: Add dropout, increase data augmentation, use early stopping
2. **Underfitting**: Increase model capacity, train longer, adjust learning rate
3. **Vanishing Gradients**: Use ReLU, batch normalization, residual connections
4. **Slow Training**: Use GPU, increase batch size, use mixed precision

### Hyperparameter Tuning
```python
# Grid search example
learning_rates = [0.001, 0.01, 0.1]
batch_sizes = [32, 64, 128]

best_accuracy = 0
best_params = {}

for lr in learning_rates:
    for bs in batch_sizes:
        # Train model with these parameters
        # Evaluate and update best_params if better
        pass
```

## 🆕 **NEW: Production Considerations**
### Model Monitoring
- **Accuracy Drift**: Monitor prediction confidence over time
- **Data Drift**: Check if input distribution changes
- **Latency**: Ensure inference time stays within limits
- **Error Rates**: Track prediction failures

### Scaling Strategies
- **Batch Processing**: Process multiple images simultaneously
- **Model Quantization**: Reduce precision for faster inference
- **Edge Deployment**: Run models on mobile/IoT devices
- **Cloud Auto-scaling**: Scale API based on traffic

### Security Best Practices
- **Input Validation**: Check image size, format, content
- **Rate Limiting**: Prevent API abuse
- **Model Poisoning**: Validate training data integrity
- **Output Sanitization**: Ensure safe prediction responses

## 🆕 **NEW: Production Considerations**
### Model Monitoring
- **Accuracy Drift**: Monitor prediction confidence over time
- **Data Drift**: Check if input distribution changes
- **Latency**: Ensure inference time stays within limits
- **Error Rates**: Track prediction failures

### Scaling Strategies
- **Batch Processing**: Process multiple images simultaneously
- **Model Quantization**: Reduce precision for faster inference
- **Edge Deployment**: Run models on mobile/IoT devices
- **Cloud Auto-scaling**: Scale API based on traffic

### Security Best Practices
- **Input Validation**: Check image size, format, content
- **Rate Limiting**: Prevent API abuse
- **Model Poisoning**: Validate training data integrity
- **Output Sanitization**: Ensure safe prediction responses

## Final Project: Complete CNN Pipeline

Combine all concepts into a complete project:

1. **Data Preparation**: Load and augment MNIST/CIFAR-10
2. **Model Building**: Implement CNN with batch normalization
3. **Training**: Train with proper monitoring and early stopping
4. **Evaluation**: Comprehensive metrics and visualization
5. **Deployment**: FastAPI web service with Docker
6. **Web Interface**: Interactive digit recognition UI
7. **Monitoring**: Performance tracking and alerting

### 🆕 **NEW: Complete Working Example**
```python
# complete_cnn_pipeline.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def create_data_loaders():
    """Create MNIST data loaders"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader

class ImprovedCNN(nn.Module):
    """Improved CNN with batch normalization"""
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def train_model():
    """Complete training pipeline"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data
    train_loader, test_loader = create_data_loaders()

    # Model
    model = ImprovedCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 5
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        accuracy = 100. * correct / total
        print(f'Epoch {epoch+1}/{epochs}: Accuracy = {accuracy:.2f}%')

    # Save model
    torch.save(model.state_dict(), 'improved_digit_classifier.pth')
    print("Model saved!")

    return model

if __name__ == '__main__':
    model = train_model()
```

This comprehensive guide takes you from understanding CNN theory to deploying production-ready models with modern best practices!
```

This comprehensive guide takes you from understanding CNN theory to deploying production-ready models with modern best practices!