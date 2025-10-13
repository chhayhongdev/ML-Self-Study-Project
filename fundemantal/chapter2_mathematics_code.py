import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

print("Chapter 2: Mathematics Behind CNNs - Code Examples")
print("=" * 60)

# 2.1 Convolution Operation
print("\n2.1 Convolution Operation")


# Manual 2D convolution
def conv2d_manual(input_matrix, kernel, stride=1, padding=0):
    """Manual 2D convolution implementation"""
    if padding > 0:
        input_matrix = np.pad(input_matrix, padding, mode='constant')

    input_h, input_w = input_matrix.shape
    kernel_h, kernel_w = kernel.shape
    output_h = (input_h - kernel_h) // stride + 1
    output_w = (input_w - kernel_w) // stride + 1

    output = np.zeros((output_h, output_w))

    for i in range(0, input_h - kernel_h + 1, stride):
        for j in range(0, input_w - kernel_w + 1, stride):
            region = input_matrix[i:i+kernel_h, j:j+kernel_w]
            output[i//stride, j//stride] = np.sum(region * kernel)

    return output


# Example
input_5x5 = np.array([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25]
])

kernel_3x3 = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])

print("Input 5x5 matrix:")
print(input_5x5)
print("\nKernel 3x3:")
print(kernel_3x3)

result = conv2d_manual(input_5x5, kernel_3x3, stride=1, padding=1)
print("\nConvolution result (with padding=1):")
print(result)

# PyTorch convolution
conv_layer = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
# Set kernel weights manually
conv_layer.weight.data = torch.tensor(kernel_3x3, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

input_tensor = torch.tensor(input_5x5, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
output_tensor = conv_layer(input_tensor)

print("\nPyTorch Conv2d result:")
print(output_tensor.squeeze().detach().numpy())

# 2.2 Pooling Operations
print("\n2.2 Pooling Operations")

# Max Pooling
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
input_pool = torch.randn(1, 1, 4, 4)
print("Input for pooling:")
print(input_pool.squeeze().numpy())

max_output = max_pool(input_pool)
print("\nMax Pooling (2x2, stride=2):")
print(max_output.squeeze().numpy())

# Average Pooling
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
avg_output = avg_pool(input_pool)
print("\nAverage Pooling (2x2, stride=2):")
print(avg_output.squeeze().numpy())

# Global Average Pooling
global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
gap_output = global_avg_pool(input_pool)
print("\nGlobal Average Pooling:")
print(gap_output.squeeze().numpy())

# 2.3 Activation Functions
print("\n2.3 Activation Functions")

x = torch.linspace(-3, 3, 100)

# Sigmoid
sigmoid = torch.sigmoid(x)
print(f"Sigmoid(-3): {torch.sigmoid(torch.tensor(-3.0)):.4f}")
print(f"Sigmoid(0): {torch.sigmoid(torch.tensor(0.0)):.4f}")
print(f"Sigmoid(3): {torch.sigmoid(torch.tensor(3.0)):.4f}")

# Tanh
tanh = torch.tanh(x)
print(f"\nTanh(-3): {torch.tanh(torch.tensor(-3.0)):.4f}")
print(f"Tanh(0): {torch.tanh(torch.tensor(0.0)):.4f}")
print(f"Tanh(3): {torch.tanh(torch.tensor(3.0)):.4f}")

# ReLU
relu = F.relu(x)
print(f"\nReLU(-1): {F.relu(torch.tensor(-1.0)):.4f}")
print(f"ReLU(0): {F.relu(torch.tensor(0.0)):.4f}")
print(f"ReLU(2): {F.relu(torch.tensor(2.0)):.4f}")

# Leaky ReLU
leaky_relu = F.leaky_relu(x, negative_slope=0.01)
print(f"\nLeaky ReLU(-1): {F.leaky_relu(torch.tensor(-1.0), 0.01):.4f}")

# ELU
elu = F.elu(x, alpha=1.0)
print(f"\nELU(-1): {F.elu(torch.tensor(-1.0), 1.0):.4f}")

# 2.4 Loss Functions
print("\n2.4 Loss Functions")

# Cross-Entropy Loss
criterion = nn.CrossEntropyLoss()

# Example: 3 samples, 5 classes
logits = torch.randn(3, 5)  # Raw outputs before softmax
targets = torch.tensor([0, 2, 4])  # True class indices

loss = criterion(logits, targets)
print(f"Cross-Entropy Loss: {loss.item():.4f}")

# Manual calculation
softmax = F.softmax(logits, dim=1)
manual_loss = -torch.mean(torch.log(softmax[range(3), targets]))
print(f"Manual Cross-Entropy: {manual_loss.item():.4f}")

# 2.5 Batch Normalization
print("\n2.5 Batch Normalization")

# Create BatchNorm layer
bn = nn.BatchNorm2d(3)  # 3 channels

# Example input: batch of 4 images, 3 channels, 32x32
input_bn = torch.randn(4, 3, 32, 32)
print(f"Input mean: {input_bn.mean():.4f}, std: {input_bn.std():.4f}")

# Apply BatchNorm
output_bn = bn(input_bn)
print(f"Output mean: {output_bn.mean():.4f}, std: {output_bn.std():.4f}")

# Check running statistics
print(f"Running mean: {bn.running_mean}")
print(f"Running var: {bn.running_var}")

# 2.6 Optimization
print("\n2.6 Optimization Algorithms")


# Simple quadratic function: f(x) = x^2
def quadratic(x):
    return x**2


def grad_quadratic(x):
    return 2*x


# Different optimizers
x_sgd = torch.tensor(5.0, requires_grad=True)
x_momentum = torch.tensor(5.0, requires_grad=True)
x_adam = torch.tensor(5.0, requires_grad=True)

optimizer_sgd = torch.optim.SGD([x_sgd], lr=0.1)
optimizer_momentum = torch.optim.SGD([x_momentum], lr=0.1, momentum=0.9)
optimizer_adam = torch.optim.Adam([x_adam], lr=0.1)

print("Optimization comparison (starting from x=5, target x=0):")
for step in range(10):
    # SGD
    optimizer_sgd.zero_grad()
    loss_sgd = quadratic(x_sgd)
    loss_sgd.backward()
    optimizer_sgd.step()

    # Momentum
    optimizer_momentum.zero_grad()
    loss_momentum = quadratic(x_momentum)
    loss_momentum.backward()
    optimizer_momentum.step()

    # Adam
    optimizer_adam.zero_grad()
    loss_adam = quadratic(x_adam)
    loss_adam.backward()
    optimizer_adam.step()

    if step % 3 == 0:
        print(".4f")

print("\nFinal values:")
print(".4f")

print("\nChapter 2 completed! Key takeaways:")
print("- Convolution extracts features with parameter sharing")
print("- Pooling reduces spatial dimensions")
print("- ReLU prevents vanishing gradients")
print("- BatchNorm stabilizes training")
print("- Adam optimizer often works best for CNNs")
