# Chapter 7: Best Practices and Troubleshooting

## 7.1 Training Best Practices

### Data Preparation
```python
# Always shuffle training data
train_loader = DataLoader(dataset, shuffle=True, ...)

# Use appropriate batch size
batch_sizes = {
    'small_dataset': 32,
    'medium_dataset': 64,
    'large_dataset': 128
}

# Normalize data properly
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
])
```

### Learning Rate Selection
```python
# Learning rate finder
def find_lr(model, train_loader, criterion, device, start_lr=1e-7, end_lr=1, num_iter=100):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=start_lr)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=(end_lr/start_lr)**(1/num_iter))

    losses = []
    lrs = []

    for i, (inputs, targets) in enumerate(train_loader):
        if i >= num_iter:
            break

        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        lrs.append(lr_scheduler.get_last_lr()[0])
        lr_scheduler.step()

    # Plot loss vs learning rate
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.show()

    return lrs, losses
```

### Early Stopping
```python
class EarlyStopping:
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

# Usage
early_stopping = EarlyStopping(patience=10, min_delta=0.001)

for epoch in range(max_epochs):
    # Training
    train_loss = train_epoch(model, train_loader, criterion, optimizer)

    # Validation
    val_loss = validate(model, val_loader, criterion)

    # Early stopping check
    if early_stopping(val_loss, model):
        print("Early stopping triggered")
        break
```

## 7.2 Common Training Issues and Solutions

### 1. Overfitting
**Symptoms:** High training accuracy, low validation accuracy
**Solutions:**
```python
# Add dropout
self.dropout = nn.Dropout(0.5)
x = self.dropout(F.relu(self.fc1(x)))

# Add data augmentation
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Use weight decay (L2 regularization)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Early stopping (see above)
```

### 2. Underfitting
**Symptoms:** Low training and validation accuracy
**Solutions:**
```python
# Increase model capacity
class DeeperModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)  # More filters
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)  # Additional layer
        # ... more layers

# Train longer
num_epochs = 100

# Reduce regularization
dropout_rate = 0.1  # Lower dropout
weight_decay = 1e-5  # Lower weight decay

# Use better optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

### 3. Vanishing/Exploding Gradients
**Symptoms:** Training stalls, NaN losses
**Solutions:**
```python
# Use Batch Normalization
self.bn1 = nn.BatchNorm2d(64)
x = self.bn1(F.relu(self.conv1(x)))

# Use ReLU instead of sigmoid/tanh
x = F.relu(self.layer(x))  # Instead of torch.sigmoid

# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Use residual connections (ResNet)
# See Chapter 4 for implementation
```

### 4. Slow Training
**Solutions:**
```python
# Use GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Increase batch size
batch_size = 128  # Larger batches for GPU efficiency

# Use mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for inputs, targets in train_loader:
    inputs, targets = inputs.to(device), targets.to(device)

    optimizer.zero_grad()

    with autocast():
        outputs = model(inputs)
        loss = criterion(outputs, targets)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## 7.3 Model Architecture Guidelines

### Convolutional Layers
```python
# Good practices
conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
# - Use kernel_size=3 (optimal receptive field)
# - Use padding=1 for same spatial size
# - Start with 64 filters, double each block

# Avoid
conv_bad = nn.Conv2d(3, 1000, 7, stride=1, padding=0)  # Too many parameters
```

### Pooling Layers
```python
# Max pooling for downsampling
pool = nn.MaxPool2d(kernel_size=2, stride=2)

# Global average pooling for classification head
gap = nn.AdaptiveAvgPool2d((1, 1))
```

### Fully Connected Layers
```python
# Progressive reduction
self.fc1 = nn.Linear(512, 256)
self.fc2 = nn.Linear(256, 128)
self.fc3 = nn.Linear(128, num_classes)

# Use dropout between FC layers
x = self.dropout(F.relu(self.fc1(x)))
```

## 7.4 Hyperparameter Tuning

### Grid Search
```python
from sklearn.model_selection import ParameterGrid

param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'batch_size': [32, 64, 128],
    'dropout': [0.3, 0.5, 0.7]
}

best_accuracy = 0
best_params = None

for params in ParameterGrid(param_grid):
    print(f"Testing params: {params}")

    # Create model with params
    model = create_model(dropout=params['dropout'])
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    train_loader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=True)

    # Quick training
    accuracy = quick_train(model, train_loader, val_loader)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_params = params

print(f"Best params: {best_params}, Accuracy: {best_accuracy}")
```

### Random Search (More Efficient)
```python
import random

def random_search(num_trials=20):
    best_accuracy = 0
    best_params = None

    for trial in range(num_trials):
        # Random hyperparameters
        lr = random.choice([0.001, 0.01, 0.1])
        batch_size = random.choice([32, 64, 128])
        dropout = random.uniform(0.1, 0.7)

        params = {'lr': lr, 'batch_size': batch_size, 'dropout': dropout}
        print(f"Trial {trial+1}: {params}")

        # Train and evaluate
        accuracy = train_with_params(params)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params

    return best_params, best_accuracy
```

## 7.5 Debugging Techniques

### Gradient Checking
```python
def check_gradients(model, input_tensor, target, epsilon=1e-7):
    """Numerical gradient checking"""
    model.eval()

    # Compute analytical gradients
    output = model(input_tensor)
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

            for i in range(param.numel()):
                # Positive perturbation
                param_flat = param.view(-1)
                param_flat[i] += epsilon
                output_pos = model(input_tensor)
                loss_pos = criterion(output_pos, target)

                # Negative perturbation
                param_flat[i] -= 2 * epsilon
                output_neg = model(input_tensor)
                loss_neg = criterion(output_neg, target)

                # Numerical gradient
                numerical_grads[name].view(-1)[i] = (loss_pos - loss_neg) / (2 * epsilon)

                # Reset parameter
                param_flat[i] += epsilon

    # Compare gradients
    for name in analytical_grads:
        diff = torch.abs(analytical_grads[name] - numerical_grads[name])
        max_diff = diff.max().item()
        print(f"{name}: Max gradient difference = {max_diff}")

        if max_diff > 1e-4:
            print(f"WARNING: Large gradient difference in {name}")
```

### Model Sanity Checks
```python
def sanity_checks(model, train_loader, val_loader, device):
    """Basic model sanity checks"""

    # 1. Overfitting to small batch
    model.train()
    inputs, targets = next(iter(train_loader))
    inputs, targets = inputs[:2].to(device), targets[:2].to(device)  # Very small batch

    for _ in range(10):  # Train on same batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    # Should achieve very low loss
    with torch.no_grad():
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        print(f"Overfitting check - Loss on small batch: {loss.item():.4f}")

    # 2. Random performance check
    model = YourModel().to(device)  # Fresh model
    accuracy = evaluate(model, val_loader)  # Should be ~10% for CIFAR-10 (random)
    print(f"Random model accuracy: {accuracy:.2f}% (should be ~10%)")

    # 3. Gradient flow check
    model.train()
    inputs, targets = next(iter(train_loader))
    inputs, targets = inputs.to(device), targets.to(device)

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()

    zero_grad_count = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm == 0:
                zero_grad_count += 1
                print(f"WARNING: Zero gradient in {name}")

    if zero_grad_count > 0:
        print(f"Found {zero_grad_count} layers with zero gradients")
```

## 7.6 Performance Optimization

### Memory Optimization
```python
# Use gradient checkpointing for large models
from torch.utils.checkpoint import checkpoint

def checkpointed_forward(model, x):
    def custom_forward(*inputs):
        return model(inputs[0])
    return checkpoint(custom_forward, x)

# Use in-place operations
x = F.relu(x, inplace=True)

# Delete unused tensors
del intermediate_tensor
torch.cuda.empty_cache()
```

### Speed Optimization
```python
# Use DataParallel for multi-GPU
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# Use cuDNN benchmark mode
torch.backends.cudnn.benchmark = True

# Use channels_last memory format
model = model.to(memory_format=torch.channels_last)
inputs = inputs.to(memory_format=torch.channels_last)
```

## 7.7 Production Considerations

### Model Versioning
```python
import json
from datetime import datetime

def save_model_with_metadata(model, accuracy, config, save_path):
    """Save model with metadata"""
    metadata = {
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat(),
        'accuracy': accuracy,
        'config': config,
        'framework': 'PyTorch',
        'pytorch_version': torch.__version__
    }

    # Save metadata
    with open(save_path + '.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    # Save model
    torch.save(model.state_dict(), save_path + '.pth')

    print(f"Model saved with metadata: {save_path}")
```

### A/B Testing Framework
```python
class ModelComparator:
    def __init__(self, model_a, model_b, test_loader):
        self.model_a = model_a
        self.model_b = model_b
        self.test_loader = test_loader

    def compare_models(self):
        """Compare two models on test set"""
        results_a = self.evaluate_model(self.model_a)
        results_b = self.evaluate_model(self.model_b)

        print("Model A Results:")
        print(f"  Accuracy: {results_a['accuracy']:.2f}%")
        print(f"  Avg Loss: {results_a['loss']:.4f}")

        print("Model B Results:")
        print(f"  Accuracy: {results_b['accuracy']:.2f}%")
        print(f"  Avg Loss: {results_b['loss']:.4f}")

        # Statistical significance test
        acc_diff = results_b['accuracy'] - results_a['accuracy']
        if abs(acc_diff) > 1.0:  # Arbitrary threshold
            winner = "B" if acc_diff > 0 else "A"
            print(f"Model {winner} significantly better!")
        else:
            print("Models perform similarly")

    def evaluate_model(self, model):
        model.eval()
        correct = 0
        total = 0
        total_loss = 0

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        return {
            'accuracy': 100. * correct / total,
            'loss': total_loss / len(self.test_loader)
        }
```

## 7.8 Final Checklist

### Before Training
- [ ] Data properly normalized
- [ ] Train/validation/test splits created
- [ ] Data augmentation appropriate for task
- [ ] Model architecture suitable for data
- [ ] Loss function correct for task
- [ ] Optimizer and learning rate chosen

### During Training
- [ ] Monitor training/validation loss
- [ ] Watch for overfitting signs
- [ ] Learning rate schedule working
- [ ] Gradients not exploding/vanishing
- [ ] Model converging

### After Training
- [ ] Evaluate on test set
- [ ] Check for data leakage
- [ ] Model performance acceptable
- [ ] Model saved with metadata
- [ ] Inference pipeline tested

### Deployment
- [ ] Model optimized for inference
- [ ] Input validation implemented
- [ ] Error handling in place
- [ ] Performance monitoring set up
- [ ] Scalability tested

---

**Congratulations!** You've completed the comprehensive CNN Masterclass curriculum. You now have the knowledge and tools to:

1. **Understand** CNN fundamentals and mathematics
2. **Build** complete CNN models from scratch
3. **Train** models with advanced techniques
4. **Evaluate** model performance thoroughly
5. **Deploy** models to production
6. **Troubleshoot** common issues
7. **Apply** CNNs to real-world problems

**Next Steps:**
- Practice with different datasets (ImageNet, custom data)
- Experiment with advanced architectures (EfficientNet, Vision Transformers)
- Deploy a model to a web service
- Contribute to open-source computer vision projects

Happy coding! 🚀