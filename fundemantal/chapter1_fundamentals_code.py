import torch
import torch.nn as nn
import torch.nn.functional as F

# Device configuration
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. Convolution Example
conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
input_tensor = torch.randn(1, 3, 32, 32)
output_tensor = conv(input_tensor)
print('Conv2d output shape:', output_tensor.shape)

# 2. Activation Function Example
x = torch.tensor([-1.0, 0.0, 2.0])
print('ReLU activation:', F.relu(x))

# 3. Pooling Example
pool = nn.MaxPool2d(kernel_size=2, stride=2)
pool_input = torch.randn(1, 16, 32, 32)
pool_output = pool(pool_input)
print('MaxPool2d output shape:', pool_output.shape)

# 4. Fully Connected Layer Example
fc = nn.Linear(16*8*8, 10)
fc_input = torch.randn(1, 16*8*8)
fc_output = fc(fc_input)
print('Linear output shape:', fc_output.shape)

# 5. Simple CNN Model
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

model = SimpleCNN()
print(model)

# 6. Minimal Training Loop (dummy data)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Dummy data: 4 images, 3 channels, 32x32, 4 labels (0-9)
dummy_inputs = torch.randn(4, 3, 32, 32)
dummy_labels = torch.randint(0, 10, (4,))

for epoch in range(2):
    optimizer.zero_grad()
    outputs = model(dummy_inputs)
    loss = criterion(outputs, dummy_labels)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

print('Finished dummy training!')
