# Chapter 1: Fundamentals of Convolutional Neural Networks (CNNs)

## 1.1 What is a Convolutional Neural Network?
A Convolutional Neural Network (CNN) is a type of deep learning model designed to process data with a grid-like topology, such as images. CNNs are inspired by the visual cortex of animals and are especially powerful for computer vision tasks.

**Key characteristics:**
- **Local connectivity:** Neurons in a layer are only connected to a small region of the previous layer.
- **Parameter sharing:** The same filter (kernel) is used across the entire input, reducing the number of parameters.
- **Hierarchical feature learning:** Lower layers learn simple features (edges, colors), while deeper layers learn complex patterns (objects, faces).

## 1.2 Why Use CNNs for Images?
- **Efficient parameter usage:** Fewer parameters than fully connected networks, making them easier to train on images.
- **Translation invariance:** Can detect features regardless of their position in the image.
- **Automatic feature extraction:** No need for manual feature engineering.

## 1.3 Core Components of a CNN

### 1.3.1 Convolutional Layer
- Applies a set of learnable filters (kernels) to the input image.
- Each filter slides (convolves) over the input, producing a feature map.

**Mathematical operation:**
$$(f * g)(x, y) = \sum_i \sum_j f(i, j) \cdot g(x-i, y-j)$$

**PyTorch Example:**
```python
import torch
import torch.nn as nn

conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
input = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
output = conv(input)
print(output.shape)  # torch.Size([1, 16, 32, 32])
```

### 1.3.2 Activation Function (ReLU)
- Introduces non-linearity.
- Most common: Rectified Linear Unit (ReLU): $f(x) = \max(0, x)$

**PyTorch Example:**
```python
import torch.nn.functional as F
x = torch.tensor([-1.0, 0.0, 2.0])
print(F.relu(x))  # tensor([0., 0., 2.])
```

### 1.3.3 Pooling Layer
- Reduces spatial dimensions (downsampling), making representations smaller and more manageable.
- Common types: Max Pooling, Average Pooling.

**PyTorch Example:**
```python
pool = nn.MaxPool2d(kernel_size=2, stride=2)
input = torch.randn(1, 16, 32, 32)
output = pool(input)
print(output.shape)  # torch.Size([1, 16, 16, 16])
```

### 1.3.4 Fully Connected Layer
- After several convolution and pooling layers, the output is flattened and passed to one or more fully connected (linear) layers for classification.

**PyTorch Example:**
```python
fc = nn.Linear(16*8*8, 10)  # Example: 16 channels, 8x8 feature map, 10 classes
input = torch.randn(1, 16*8*8)
output = fc(input)
print(output.shape)  # torch.Size([1, 10])
```

## 1.4 Building a Simple CNN in PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)      # 3 input channels (RGB), 6 output, 5x5 kernel
        self.pool = nn.MaxPool2d(2, 2)       # 2x2 max pooling
        self.conv2 = nn.Conv2d(6, 16, 5)     # 6 input, 16 output, 5x5 kernel
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)         # 10 output classes (e.g., CIFAR-10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # Conv1 + ReLU + Pool
        x = self.pool(F.relu(self.conv2(x))) # Conv2 + ReLU + Pool
        x = x.view(-1, 16 * 5 * 5)           # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Instantiate and print model
model = SimpleCNN()
print(model)
```

## 1.5 Training a CNN: The Big Picture

1. **Prepare data:** Load and preprocess images (normalize, augment).
2. **Define model:** Build the CNN architecture.
3. **Set loss and optimizer:** E.g., CrossEntropyLoss and SGD/Adam.
4. **Train:** Loop over data, forward pass, compute loss, backward pass, update weights.
5. **Evaluate:** Test on validation/test set.

**Minimal training loop example:**
```python
import torch.optim as optim

# Assume trainloader is a DataLoader for your dataset
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:    # print every 100 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0
print('Finished Training')
```

## 1.6 Visualizing CNN Operations

### Convolution Example (Manual Calculation)
Suppose you have a 3x3 input and a 2x2 kernel:

Input:
```
1 2 3
4 5 6
7 8 9
```
Kernel:
```
1 0
0 -1
```
Convolution output (top-left position):
$$(1*1 + 2*0 + 4*0 + 5*(-1)) = 1 + 0 + 0 - 5 = -4$$

### Pooling Example
Max pooling with 2x2 window on:
```
1 3
2 4
```
Result: max(1,3,2,4) = 4

## 1.7 Summary
- CNNs are the backbone of modern computer vision.
- They use convolution, activation, pooling, and fully connected layers to learn from images.
- PyTorch makes it easy to build and train CNNs with just a few lines of code.

---

**Next:** Dive deeper into the mathematics of convolution and backpropagation in Chapter 2!
