# CNN Masterclass Project

A complete Convolutional Neur## 💾 Model Saving & Loadingl Network (CNN) project implementing image classification on CIFAR-10 dataset, featuring ResNet18 architecture with advanced techniques.

## 📁 Project Structure

```
CNN_Masterclass_Project/
├── copy_of_cnn_masterclass.py    # Main training script with ResNet18
├── cnn_cif## 📊 Model Performancer10.pth             # Trained model weights (44.8MB) - Final model
├── best_cifar10_val.pth        # Best validation model (may not exist)
├── model_inference.py           # Inference script for predictions
├── CNN_Masterclass_Lessons.md   # Comprehensive lessons (theory to deployment)
└── README.md                    # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- PyTorch with MPS support (macOS M2/M3)
- Required packages: torch, torchvision, numpy, matplotlib, scikit-learn, pillow

### Setup Environment
```bash
# Create and activate conda environment
conda create -n cnn_env python=3.9
conda activate cnn_env

# Install dependencies
conda install pytorch torchvision torchaudio -c pytorch
conda install numpy matplotlib scikit-learn pillow
```

### Training the Model
```bash
cd CNN_Masterclass_Project
python copy_of_cnn_masterclass.py
```
- Trains ResNet18 on CIFAR-10
- Uses data augmentation and batch normalization
- Saves model as `cnn_cifar10.pth`

### Making Predictions
```bash
python model_inference.py
```

Or in Python:
```python
from model_inference import load_model, predict_from_url

model = load_model()
predict_from_url(model, 'https://example.com/image.jpg')
```

## � Model Saving & Loading

### Automatic Model Saving During Training

The training script automatically saves your model in two ways:

```python
# Best validation model saving (happens when validation accuracy improves)
best_path = 'best_cifar10_val.pth'
if val_acc > best_val_acc:
    best_val_acc = val_acc
    torch.save(model.state_dict(), best_path)
    print(f"💾 Best validation model saved! Accuracy: {val_acc:.2f}%")

# Final model saving (always happens at end of training)
PATH = 'cnn_cifar10.pth'
torch.save(model.state_dict(), PATH)
print("💾 Final model saved!")
```

**What gets saved:**
- `cnn_cifar10.pth` - Final trained model weights (always saved)
- `best_cifar10_val.pth` - Best model based on validation accuracy (only if validation accuracy improved during training)

### Choosing Which Model to Use

- **For production/deployment**: Use `cnn_cifar10.pth` (final model, always available)
- **For best performance**: Use `best_cifar10_val.pth` if it exists (best validation accuracy), otherwise use `cnn_cifar10.pth`
- **For continued training**: Load `cnn_cifar10.pth` and resume training

### Model Saving Methods

#### 1. State Dict Only (Recommended)
```python
# Save only the model's learned parameters
torch.save(model.state_dict(), 'model_weights.pth')

# Load
model = ResNet18()  # Create model architecture
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()
```

#### 2. Full Model Saving
```python
# Save entire model (includes architecture)
torch.save(model, 'full_model.pth')

# Load
model = torch.load('full_model.pth')
model.eval()
```

#### 3. Checkpoint Saving (Training Resume)
```python
# Save complete training state
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'loss': loss,
    'accuracy': accuracy
}
torch.save(checkpoint, 'checkpoint.pth')

# Load and resume training
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
start_epoch = checkpoint['epoch']
```

### Model Loading for Inference

#### Basic Loading
```python
import torch
from torchvision.models import resnet18

def load_model(model_path='cnn_cifar10.pth'):
    # Create model architecture (must match training exactly)
    model = resnet18(num_classes=10)

    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location='cpu'))

    # Set to evaluation mode
    model.eval()

    return model

# Load the final trained model
model = load_model('cnn_cifar10.pth')

# Or load the best validation model
best_model = load_model('best_cifar10_val.pth')
```

#### GPU Loading (if available)
```python
def load_model_gpu(model_path='cnn_cifar10.pth'):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    model = resnet18(num_classes=10)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model, device
```

### Using Saved Models for Predictions

#### Single Image Prediction
```python
import torch
from PIL import Image
import torchvision.transforms as transforms

def predict_single_image(model, image_path, device='cpu'):
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')

    # Define preprocessing (same as training)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # CIFAR-10 size
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Preprocess
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    # CIFAR-10 class names
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    return classes[predicted_class.item()], confidence.item()
```

#### Batch Predictions
```python
def predict_batch(model, image_paths, device='cpu'):
    # Load all images
    images = []
    for path in image_paths:
        img = Image.open(path).convert('RGB')
        images.append(img)

    # Batch preprocessing
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Create batch tensor
    batch_tensors = torch.stack([transform(img) for img in images]).to(device)

    # Batch prediction
    with torch.no_grad():
        outputs = model(batch_tensors)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidences, predicted_classes = torch.max(probabilities, 1)

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    results = []
    for i, (pred_class, conf) in enumerate(zip(predicted_classes, confidences)):
        results.append({
            'image': image_paths[i],
            'prediction': classes[pred_class.item()],
            'confidence': conf.item()
        })

    return results
```

### Model Deployment Best Practices

#### 1. Model Optimization
```python
# Convert to TorchScript for faster inference
model = load_model()
model.eval()

# Trace the model
example_input = torch.randn(1, 3, 32, 32)
traced_model = torch.jit.trace(model, example_input)
torch.jit.save(traced_model, 'model_traced.pt')

# Load traced model
model = torch.jit.load('model_traced.pt')
```

#### 2. Quantization (Reduce Model Size)
```python
# Dynamic quantization
model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
torch.save(model, 'model_quantized.pth')
```

#### 3. ONNX Export (Cross-Platform)
```python
# Export to ONNX format
torch.onnx.export(
    model,
    torch.randn(1, 3, 32, 32),
    'model.onnx',
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
```

### Model Versioning & Management

#### Save with Metadata
```python
def save_model_with_metadata(model, accuracy, epoch, filename):
    metadata = {
        'model_state_dict': model.state_dict(),
        'accuracy': accuracy,
        'epoch': epoch,
        'timestamp': str(datetime.now()),
        'pytorch_version': torch.__version__,
        'architecture': 'ResNet18',
        'dataset': 'CIFAR-10',
        'classes': ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    }
    torch.save(metadata, filename)
    print(f"Model saved with metadata: {filename}")
```

#### Load with Validation
```python
def load_model_with_validation(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")

    try:
        checkpoint = torch.load(filepath, map_location='cpu')

        # Validate checkpoint structure
        required_keys = ['model_state_dict', 'accuracy', 'epoch']
        for key in required_keys:
            if key not in checkpoint:
                raise ValueError(f"Invalid checkpoint: missing {key}")

        # Recreate model
        model = resnet18(num_classes=10)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        print(f"Model loaded successfully!")
        print(f"Accuracy: {checkpoint.get('accuracy', 'N/A')}")
        print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")

        return model, checkpoint

    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")
```

### Common Model Saving Issues & Solutions

#### Issue 1: CUDA Runtime Error
```python
# Problem: Model saved on GPU, loaded on CPU
# Solution: Use map_location
model.load_state_dict(torch.load('model.pth', map_location='cpu'))
```

#### Issue 2: Architecture Mismatch
```python
# Problem: Model architecture doesn't match saved weights
# Solution: Ensure exact same model definition
model = ResNet18()  # Must match training architecture exactly
model.load_state_dict(torch.load('model.pth'))
```

#### Issue 3: Memory Issues During Loading
```python
# Problem: Large model causes memory error
# Solution: Load in evaluation mode and use CPU if needed
model = resnet18(num_classes=10)
model.load_state_dict(torch.load('model.pth', map_location='cpu'))
model.eval()  # Reduces memory usage
```

## �📊 Model Performance

- **Architecture**: ResNet18 with residual blocks
- **Test Accuracy**: 83.09%
- **Training Time**: ~47 epochs with early stopping
- **Classes**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

### Per-Class Accuracy
- Car: 93.1% ⭐
- Ship: 91.6% ⭐
- Truck: 89.8% ⭐
- Cat: 65.8% (weakest - common CIFAR-10 challenge)

## 🎯 Key Features

### Training Features
- ✅ ResNet18 architecture with residual connections
- ✅ Batch normalization for stable training
- ✅ Data augmentation (crop, flip, rotation, color jitter)
- ✅ Early stopping with patience
- ✅ Learning rate scheduling
- ✅ GPU acceleration (MPS on macOS M2)

### Inference Features
- ✅ Load trained model
- ✅ Predict from URLs or local files
- ✅ Confidence scores and probabilities
- ✅ Image preprocessing pipeline
- ✅ Error handling

## 📚 Learning Resources

The `CNN_Masterclass_Lessons.md` contains 7 comprehensive lessons:

1. **CNN Theory & Basics** - Neural network fundamentals
2. **Mathematics Behind CNNs** - Convolution, pooling, backpropagation
3. **Building Your First CNN** - PyTorch implementation
4. **Advanced Techniques** - Batch norm, data augmentation, ResNet
5. **Model Evaluation** - Metrics, confusion matrices
6. **Model Deployment** - Saving/loading, inference
7. **Best Practices** - Optimization and troubleshooting

## 🔧 Training Customization: Parts You Can Tweak

The training script has **20+ configurable parameters** across multiple categories. Here's a comprehensive guide to tweaking different parts for better performance:

## 🔧 Training Customization: Parts You Can Tweak

The training script has **20+ configurable parameters** across multiple categories. Here's a comprehensive guide to tweaking different parts for better performance:

| Category | Parameters | Current Values | Impact Level |
|----------|------------|----------------|--------------|
| **Data** | Augmentation, normalization | RandomCrop, CIFAR-10 stats | High |
| **Model** | Architecture, depth | ResNet18, 4 conv blocks | Very High |
| **Training** | Epochs, batch size | 50 epochs, 128 batch | High |
| **Optimizer** | Type, momentum, weight decay | SGD, 0.9 momentum, 1e-4 wd | High |
| **Scheduler** | Type, patience | ReduceLROnPlateau | Medium |
| **Regularization** | Dropout, early stopping | 0.5 dropout, patience=10 | Medium |

#### Current Configuration:
```python
# Data augmentation transforms
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),          # Random 32x32 crop with 4px padding
    transforms.RandomHorizontalFlip(),             # 50% chance horizontal flip
    transforms.RandomRotation(15),                 # ±15° rotation
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # Color variations
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))     # CIFAR-10 stats
])
```

#### Tweak Options:
- **More augmentation**: Add `transforms.RandomAffine()`, `transforms.GaussianBlur()`
- **Less augmentation**: Remove rotation/color jitter for faster convergence
- **Advanced augmentation**: Add `transforms.AutoAugment()` or `transforms.RandAugment()`
- **Different normalization**: Try ImageNet stats `(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)`

### 2. 🏗️ Model Architecture

#### Current: ResNet18 with CustomCNN fallback
```python
# Available architectures
def ResNet18(): return ResNet(BasicBlock, [2, 2, 2, 2])    # Current choice
def ResNet34(): return ResNet(BasicBlock, [3, 4, 6, 3])    # Deeper ResNet
def ResNet50(): return ResNet(Bottleneck, [3, 4, 6, 3])    # Bottleneck blocks

# Custom CNN architecture
class CustomCNN(nn.Module):
    def __init__(self):
        # 4 conv layers: 32→64→128→256 channels
        # BatchNorm after each conv
        # Dropout 0.5 before FC layers
        # FC: 1024→512→10
```

#### Tweak Options:
- **Deeper networks**: Switch to ResNet34/ResNet50 for potentially higher accuracy
- **Wider networks**: Increase channel dimensions (32→64→128→512)
- **Different activations**: Try `nn.SiLU()`, `nn.Mish()` instead of `nn.ReLU()`
- **Skip connections**: Add residual connections to CustomCNN
- **Attention mechanisms**: Add SE blocks or CBAM to ResNet blocks

### 3. ⚙️ Core Training Hyperparameters

#### Current Settings:
```python
num_epochs     = 50      # Total training epochs
learning_rate  = 0.01    # Initial learning rate
momentum       = 0.9     # SGD momentum
batch_size     = 128     # Samples per batch
```

#### Tweak Options:
- **Epochs**: `30-100` (more epochs for convergence, but watch overfitting)
- **Learning rate**: `0.001-0.1` (lower for stability, higher for faster learning)
- **Batch size**: `64-256` (larger batches = more stable gradients)
- **Momentum**: `0.8-0.95` (higher = smoother optimization)

### 4. 🎯 Optimizer Selection

#### Current: SGD with momentum + weight decay
```python
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=1e-4)
```

#### Alternative Optimizers:
```python
# Adam optimizer (often converges faster)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# AdamW (better weight decay handling)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

# RMSprop (alternative momentum-based)
optimizer = optim.RMSprop(model.parameters(), lr=0.001, momentum=0.9)
```

### 5. 📉 Learning Rate Scheduling

#### Current: ReduceLROnPlateau
```python
scheduler_type = 'plateau'  # Options: 'plateau', 'step', None
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=2, min_lr=1e-6
)
```

#### Alternative Schedulers:
```python
# Step scheduler (reduce LR every N epochs)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Cosine annealing (smooth LR decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

# OneCycleLR (aggressive LR cycling)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, epochs=50, steps_per_epoch=len(train_loader))

# Exponential decay
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
```

### 6. 🛡️ Regularization Techniques

#### Current: Weight decay + Dropout
```python
# In optimizer
weight_decay=1e-4

# In model
self.dropout = nn.Dropout(0.5)
```

#### Additional Regularization:
```python
# L1 regularization (add to loss)
l1_lambda = 1e-5
l1_norm = sum(p.abs().sum() for p in model.parameters())
loss = criterion(outputs, labels) + l1_lambda * l1_norm

# Label smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 7. 🎯 Loss Functions

#### Current: CrossEntropyLoss
```python
criterion = nn.CrossEntropyLoss()
```

#### Alternative Losses:
```python
# Focal Loss (for imbalanced classes)
# Requires installing torchvision or implementing manually

# Label smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Weighted loss (if classes are imbalanced)
class_weights = torch.tensor([1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # Boost cat class
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

### 8. ⏹️ Early Stopping Configuration

#### Current Settings:
```python
es_patience    = 10     # Wait 10 epochs for improvement
es_min_delta   = 1e-4   # Minimum loss improvement threshold
```

#### Tweak Options:
- **More patient**: `es_patience = 15-20` (let model train longer)
- **Less patient**: `es_patience = 5-7` (stop earlier to save time)
- **Stricter threshold**: `es_min_delta = 1e-5` (require smaller improvements)
- **Looser threshold**: `es_min_delta = 1e-3` (accept larger improvements)

### 9. 💾 Data Loading Parameters

#### Current Configuration:
```python
batch_size = 128
train_loader = DataLoader(..., batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
```

#### Tweak Options:
- **Batch size**: `64, 128, 256, 512` (trade-off: memory vs stability)
- **Workers**: `num_workers=2-4` on multi-core systems (faster loading)
- **Pin memory**: `pin_memory=False` if experiencing CUDA issues
- **Shuffle**: `shuffle=False` for deterministic results

### 10. 📈 Training Loop Modifications

#### Current: Standard training with validation
```python
for epoch in range(1, num_epochs+1):
    tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc = run_epoch(model, val_loader, criterion, None)
    # ... logging and early stopping
```

#### Advanced Training Techniques:
```python
# Gradient accumulation (for larger effective batch sizes)
accumulation_steps = 4
for i, (images, labels) in enumerate(train_loader):
    outputs = model(images)
    loss = criterion(outputs, labels) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# Mixed precision training (faster, less memory)
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
with autocast():
    outputs = model(images)
    loss = criterion(outputs, labels)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 11. 📊 Validation Strategy

#### Current: 10% of training data for validation
```python
val_split = 0.1  # 10% of training data
```

#### Alternatives:
- **Cross-validation**: K-fold validation instead of single split
- **Larger validation**: `val_split = 0.2` for more reliable metrics
- **Smaller validation**: `val_split = 0.05` for more training data

### 12. 🔍 Monitoring & Logging

#### Current: Basic console output
```python
print(f"Epoch {epoch:02d}/{num_epochs} | "
      f"train: loss={tr_loss:.4f}, acc={tr_acc*100:.2f}% | "
      f"val: loss={val_loss:.4f}, acc={val_acc*100:.2f}% | "
      f"lr={get_lr(optimizer):.5f}")
```

#### Enhanced Monitoring:
```python
# TensorBoard logging
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
writer.add_scalars('Loss', {'train': tr_loss, 'val': val_loss}, epoch)
writer.add_scalars('Accuracy', {'train': tr_acc, 'val': val_acc}, epoch)

# Weights & Biases (wandb)
import wandb
wandb.log({'train_loss': tr_loss, 'val_loss': val_loss, 'train_acc': tr_acc, 'val_acc': val_acc})
```

## 🎯 Quick Performance Tweaks

### For Faster Convergence:
1. **Switch to Adam**: `optim.Adam(model.parameters(), lr=0.001)`
2. **Increase batch size**: `batch_size = 256`
3. **Add scheduler**: `OneCycleLR` or `CosineAnnealingLR`
4. **Reduce epochs**: Focus on `es_patience` for early stopping

### For Higher Accuracy:
1. **Deeper model**: Switch to `ResNet34()` or `ResNet50()`
2. **More augmentation**: Add `AutoAugment` or `RandAugment`
3. **Increase epochs**: Set `num_epochs = 100`
4. **Label smoothing**: `label_smoothing=0.1`
5. **Larger batch size**: `batch_size = 256`

### For Better Generalization:
1. **More dropout**: Increase to `nn.Dropout(0.3-0.5)`
2. **Weight decay**: Increase to `weight_decay=5e-4`
3. **Data augmentation**: Add more transforms
4. **Early stopping**: Be more patient with `es_patience=15`

### For Debugging Training Issues:
1. **Overfitting**: Increase regularization, add augmentation
2. **Underfitting**: Increase model capacity, train longer
3. **Unstable loss**: Reduce learning rate, increase batch size
4. **Slow convergence**: Switch optimizer, adjust scheduler

## 📋 Tweak Priority (Most Impact First)

1. **Model Architecture** → Biggest accuracy impact
2. **Optimizer Choice** → Affects convergence speed
3. **Learning Rate** → Critical for stable training
4. **Batch Size** → Memory vs stability trade-off
5. **Data Augmentation** → Prevents overfitting
6. **Regularization** → Controls overfitting
7. **Scheduler** → Optimizes learning dynamics
8. **Early Stopping** → Prevents wasted computation

## 🧪 Experiment Tracking

Keep track of your experiments:
```python
experiment_config = {
    'model': 'ResNet18',
    'optimizer': 'SGD',
    'lr': 0.01,
    'batch_size': 128,
    'augmentation': 'basic',
    'scheduler': 'plateau',
    'final_accuracy': 83.09
}
```

Remember: **Always change one thing at a time** when experimenting to understand what improves performance!

### Train a New Model
```python
# Modify hyperparameters in copy_of_cnn_masterclass.py
num_epochs = 100  # Increase for better accuracy
batch_size = 256  # Larger batch for stability

# Run training
python copy_of_cnn_masterclass.py
```

### Custom Inference
```python
from model_inference import load_model, predict_from_file
import os

# Load the best model if it exists, otherwise use final model
if os.path.exists('best_cifar10_val.pth'):
    model = load_model('best_cifar10_val.pth')
    print("Using best validation model")
else:
    model = load_model('cnn_cifar10.pth')
    print("Using final trained model")

# Predict on your image
result = predict_from_file(model, 'my_image.jpg')
print(f"Predicted: {result[0]}, Confidence: {result[1]:.2f}")
```

### Batch Predictions
```python
from model_inference import load_model, predict_from_url

model = load_model()
urls = [
    'https://example.com/image1.jpg',
    'https://example.com/image2.jpg'
]

for url in urls:
    predict_from_url(model, url, show_image=False)
```

## 🛠️ Customization

### Modify Architecture
Edit `copy_of_cnn_masterclass.py`:
```python
# Change to different ResNet variant
def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])

model = ResNet50().to(device)
```

### Adjust Hyperparameters
```python
# In the hyperparameters section
learning_rate = 0.001  # Lower for more stable training
weight_decay = 5e-4    # Higher for more regularization
batch_size = 128       # Adjust based on GPU memory
```

### Add More Augmentation
```python
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),  # Increase rotation
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # More color variation
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
```

## 📈 Improving Performance

### Quick Wins
1. **Increase epochs**: Set `num_epochs = 100`
2. **Larger batch size**: `batch_size = 256`
3. **More augmentation**: Add Cutout or MixUp
4. **Better optimizer**: Try AdamW instead of SGD

### Advanced Techniques
1. **Learning rate finder**: Use lr_finder to find optimal LR
2. **Progressive resizing**: Train on smaller images first
3. **Label smoothing**: For better generalization
4. **Model ensembling**: Combine multiple models

## 🔍 Troubleshooting

### Common Issues
- **Out of memory**: Reduce batch size
- **Low accuracy**: Increase epochs, check data preprocessing
- **Model not loading**: Ensure correct file path
- **Import errors**: Activate conda environment

### Performance Tips
- Use MPS acceleration on M2/M3 Macs
- Monitor GPU memory usage
- Use `torch.cuda.empty_cache()` if needed
- Profile with `torch.profiler`

## 📝 Notes

- Model trained on CIFAR-10 (32x32 images)
- Best performance on similar-sized images
- Preprocessing normalizes images to [-1, 1] range
- Model expects RGB images

## 🤝 Contributing

Feel free to:
- Experiment with different architectures
- Add more augmentation techniques
- Implement advanced training methods
- Share your improvements!

---

**Created**: October 11, 2025
**Model**: ResNet18
**Dataset**: CIFAR-10
**Accuracy**: 83.09%

Happy coding! 🚀# ML-Self-Learining
