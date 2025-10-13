# Chapter 10: Edge Deployment and Model Optimization

## 10.1 Introduction to Edge Computing

Edge computing refers to processing data near the source of data generation rather than relying on centralized cloud computing. For computer vision models, this means deploying CNNs on edge devices like mobile phones, IoT devices, embedded systems, and edge servers.

### Key Challenges for Edge Deployment:
1. **Limited computational resources** - CPUs, GPUs, memory constraints
2. **Power consumption** - Battery life considerations
3. **Network connectivity** - Offline operation requirements
4. **Real-time performance** - Low latency requirements
5. **Model size** - Storage and memory limitations

### Edge Deployment Strategies:
1. **Model Optimization** - Reduce model size and complexity
2. **Quantization** - Reduce numerical precision
3. **Pruning** - Remove unnecessary parameters
4. **Knowledge Distillation** - Transfer knowledge to smaller models
5. **Neural Architecture Search** - Find optimal architectures for edge

## 10.2 Model Quantization

Quantization reduces the precision of model weights and activations from 32-bit floating point to lower precision formats (16-bit, 8-bit, or even 4-bit).

### 10.2.1 Post-Training Quantization

```python
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.quantization import QuantStub, DeQuantStub
import torch.quantization as quant

class QuantizableEfficientNet(nn.Module):
    """EfficientNet with quantization support"""

    def __init__(self, num_classes=10):
        super(QuantizableEfficientNet, self).__init__()

        # Quantization stubs
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        # Simplified EfficientNet-like architecture
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6()
        )

        self.blocks = nn.Sequential(
            self._make_block(32, 16, 1, 1),
            self._make_block(16, 24, 6, 2),
            self._make_block(24, 40, 6, 2),
            self._make_block(40, 80, 6, 2),
            self._make_block(80, 112, 6, 1),
            self._make_block(112, 160, 6, 2),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(160, num_classes)
        )

    def _make_block(self, in_channels, out_channels, expand_ratio, stride):
        """Create a MBConv-like block"""
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels * expand_ratio, 1, bias=False),
            nn.BatchNorm2d(in_channels * expand_ratio),
            nn.ReLU6(),
            nn.Conv2d(in_channels * expand_ratio, in_channels * expand_ratio, 3,
                     stride=stride, padding=1, groups=in_channels * expand_ratio, bias=False),
            nn.BatchNorm2d(in_channels * expand_ratio),
            nn.ReLU6(),
            nn.Conv2d(in_channels * expand_ratio, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x = self.quant(x)  # Quantize input
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.dequant(x)  # Dequantize output
        return x

    def fuse_model(self):
        """Fuse Conv2d + BatchNorm2d + ReLU layers for quantization"""
        for module in self.modules():
            if isinstance(module, nn.Sequential):
                for i in range(len(module) - 2):
                    # Fuse Conv2d + BatchNorm2d + ReLU6
                    if (isinstance(module[i], nn.Conv2d) and
                        isinstance(module[i+1], nn.BatchNorm2d) and
                        isinstance(module[i+2], nn.ReLU6)):
                        torch.quantization.fuse_modules(
                            module, [str(i), str(i+1), str(i+2)], inplace=True
                        )

def quantize_model(model, calibration_loader):
    """Post-training quantization"""

    # Set model to evaluation mode
    model.eval()

    # Fuse layers
    model.fuse_model()

    # Specify quantization configuration
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

    # Prepare model for quantization
    torch.quantization.prepare(model, inplace=True)

    # Calibrate with representative data
    print("Calibrating quantization...")
    with torch.no_grad():
        for images, _ in calibration_loader:
            model(images)
            break  # Use just one batch for calibration

    # Convert to quantized model
    torch.quantization.convert(model, inplace=True)

    return model

def compare_model_sizes(original_model, quantized_model):
    """Compare original vs quantized model sizes"""

    # Save original model
    torch.save(original_model.state_dict(), 'models/original_model.pth')
    original_size = os.path.getsize('models/original_model.pth') / (1024 * 1024)  # MB

    # Save quantized model
    torch.save(quantized_model.state_dict(), 'models/quantized_model.pth')
    quantized_size = os.path.getsize('models/quantized_model.pth') / (1024 * 1024)  # MB

    print(".2f")
    print(".2f")
    print(".1f")

    return original_size, quantized_size
```

### 10.2.2 Quantization-Aware Training (QAT)

```python
def quantize_aware_training(model, train_loader, val_loader, epochs=5):
    """Quantization-aware training"""

    # Fuse layers
    model.fuse_model()

    # Set quantization config
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')

    # Prepare for QAT
    model = torch.quantization.prepare_qat(model, inplace=True)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print("Starting Quantization-Aware Training...")

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        accuracy = 100. * correct / total
        print(f"QAT Epoch {epoch+1}/{epochs}: {accuracy:.2f}%")

    # Convert to quantized model
    model = torch.quantization.convert(model, inplace=True)

    return model
```

## 10.3 Model Pruning

Pruning removes unnecessary weights or neurons from the model to reduce size and computation.

### 10.3.1 Weight Pruning

```python
import torch.nn.utils.prune as prune

def apply_weight_pruning(model, pruning_rate=0.3):
    """Apply weight pruning to convolutional layers"""

    print(f"Applying {pruning_rate*100}% weight pruning...")

    # Prune convolutional layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=pruning_rate)
            print(f"Pruned {name}: {prune.sparsity(module, 'weight'):.1%} sparsity")

    return model

def apply_structured_pruning(model, pruning_rate=0.3):
    """Apply structured pruning (remove entire channels)"""

    print(f"Applying {pruning_rate*100}% structured pruning...")

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(module, name='weight', amount=pruning_rate, n=2, dim=0)
            print(f"Structured pruned {name}: {prune.sparsity(module, 'weight'):.1%} sparsity")

    return model

def remove_pruning_masks(model):
    """Remove pruning masks and make pruning permanent"""

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.remove(module, 'weight')

    print("Pruning masks removed - pruning is now permanent")
    return model

def iterative_pruning_finetuning(model, train_loader, val_loader,
                                pruning_rate=0.1, iterations=3):
    """Iterative pruning with fine-tuning"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)  # Lower LR for fine-tuning

    for iteration in range(iterations):
        print(f"\\nPruning Iteration {iteration + 1}/{iterations}")

        # Apply pruning
        model = apply_weight_pruning(model, pruning_rate)

        # Fine-tune
        model.train()
        for epoch in range(2):  # Short fine-tuning
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
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        accuracy = 100. * correct / total
        print(f"Iteration {iteration + 1} accuracy: {accuracy:.2f}%")

    # Remove pruning masks
    model = remove_pruning_masks(model)

    return model
```

## 10.4 Knowledge Distillation

Knowledge distillation transfers knowledge from a large "teacher" model to a smaller "student" model.

### 10.4.1 Teacher-Student Training

```python
class DistillationLoss(nn.Module):
    """Knowledge distillation loss"""

    def __init__(self, temperature=3.0, alpha=0.7):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, student_outputs, teacher_outputs, labels):
        # Hard loss (cross-entropy with ground truth)
        hard_loss = self.criterion(student_outputs, labels)

        # Soft loss (KL divergence with teacher predictions)
        student_soft = F.log_softmax(student_outputs / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_outputs / self.temperature, dim=1)
        soft_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (self.temperature ** 2)

        # Combined loss
        loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss

        return loss

def distill_knowledge(teacher_model, student_model, train_loader, val_loader,
                     temperature=3.0, alpha=0.7, epochs=10):
    """Knowledge distillation training"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    teacher_model = teacher_model.to(device).eval()  # Teacher in eval mode
    student_model = student_model.to(device).train()

    distillation_loss = DistillationLoss(temperature, alpha)
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-3)

    print("Starting Knowledge Distillation...")
    print(f"Teacher parameters: {sum(p.numel() for p in teacher_model.parameters()):,}")
    print(f"Student parameters: {sum(p.numel() for p in student_model.parameters()):,}")

    for epoch in range(epochs):
        student_model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # Get teacher predictions
            with torch.no_grad():
                teacher_outputs = teacher_model(images)

            # Get student predictions
            student_outputs = student_model(images)

            # Compute distillation loss
            loss = distillation_loss(student_outputs, teacher_outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation
        student_model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = student_model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        accuracy = 100. * correct / total
        avg_loss = running_loss / len(train_loader)

        print(f"Epoch {epoch+1}/{epochs}: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    return student_model
```

## 10.5 ONNX and TensorRT Deployment

### 10.5.1 Export to ONNX

```python
import onnxruntime as ort
import onnx
from onnx import numpy_helper

def export_to_onnx(model, input_shape=(1, 3, 224, 224), onnx_path='model.onnx'):
    """Export PyTorch model to ONNX format"""

    # Create dummy input
    dummy_input = torch.randn(input_shape)

    # Set model to evaluation mode
    model.eval()

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    print(f"Model exported to {onnx_path}")

    # Verify ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model validation passed")

    return onnx_path

def onnx_inference(onnx_path, input_tensor):
    """Run inference with ONNX model"""

    # Create ONNX Runtime session
    ort_session = ort.InferenceSession(onnx_path)

    # Prepare input
    ort_inputs = {ort_session.get_inputs()[0].name: input_tensor.numpy()}

    # Run inference
    ort_outputs = ort_session.run(None, ort_inputs)

    return torch.from_numpy(ort_outputs[0])

def compare_pytorch_onnx_inference(pytorch_model, onnx_path, test_loader):
    """Compare PyTorch and ONNX model outputs"""

    pytorch_model.eval()
    device = next(pytorch_model.parameters()).device

    print("Comparing PyTorch vs ONNX inference...")

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            # PyTorch inference
            pytorch_output = pytorch_model(images)

            # ONNX inference
            onnx_output = onnx_inference(onnx_path, images)

            # Compare outputs
            max_diff = torch.max(torch.abs(pytorch_output - onnx_output))
            mean_diff = torch.mean(torch.abs(pytorch_output - onnx_output))

            print(f"Max difference: {max_diff:.6f}")
            print(f"Mean difference: {mean_diff:.6f}")

            # Check if outputs are close enough
            if max_diff < 1e-4:
                print("✓ PyTorch and ONNX outputs match!")
            else:
                print("⚠ PyTorch and ONNX outputs differ significantly")

            break  # Test only one batch
```

### 10.5.2 TensorRT Optimization (NVIDIA GPUs)

```python
try:
    import tensorrt as trt
    from torch2trt import torch2trt, TRTModule

    def convert_to_tensorrt(model, input_shape=(1, 3, 224, 224), fp16_mode=True):
        """Convert PyTorch model to TensorRT"""

        device = torch.device('cuda')

        # Create dummy input
        dummy_input = torch.randn(input_shape).to(device)
        model = model.to(device).eval()

        # Convert to TensorRT
        if fp16_mode:
            model_trt = torch2trt(model, [dummy_input], fp16_mode=True)
        else:
            model_trt = torch2trt(model, [dummy_input])

        print("Model converted to TensorRT")

        return model_trt

    def tensorrt_inference_speed_test(model_trt, test_loader, num_runs=100):
        """Test TensorRT inference speed"""

        device = torch.device('cuda')
        model_trt = model_trt.to(device)

        # Warm up
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)
                _ = model_trt(images)
                break

        # Timing
        import time
        start_time = time.time()

        with torch.no_grad():
            for i, (images, _) in enumerate(test_loader):
                if i >= num_runs:
                    break
                images = images.to(device)
                _ = model_trt(images)

        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / num_runs

        print(f"TensorRT inference: {avg_time:.4f} seconds per batch")
        print(f"Throughput: {1/avg_time:.2f} batches/second")

        return avg_time

except ImportError:
    print("TensorRT not available. Install with: pip install torch2trt tensorrt")

    def convert_to_tensorrt(*args, **kwargs):
        raise NotImplementedError("TensorRT not available")

    def tensorrt_inference_speed_test(*args, **kwargs):
        raise NotImplementedError("TensorRT not available")
```

## 10.6 Mobile Deployment

### 10.6.1 PyTorch Mobile

```python
def optimize_for_mobile(model):
    """Optimize model for mobile deployment"""

    # Set to evaluation mode
    model.eval()

    # Fuse operations
    if hasattr(model, 'fuse_model'):
        model.fuse_model()

    # Convert to TorchScript
    scripted_model = torch.jit.script(model)

    # Save for mobile
    scripted_model.save("models/mobile_model.pt")

    print("Model optimized and saved for mobile deployment")

    return scripted_model

def create_mobile_app_structure():
    """Create mobile app structure for iOS/Android"""

    mobile_app_structure = """
Mobile App Structure:
├── ios/
│   ├── ModelLoader.swift
│   ├── VisionProcessor.mm
│   └── TorchModule.h
├── android/
│   ├── ModelLoader.kt
│   ├── VisionProcessor.java
│   └── native-lib.cpp
├── models/
│   ├── optimized_model.pt
│   └── model_metadata.json
└── shared/
    ├── preprocessing.py
    └── postprocessing.py
"""

    print("Mobile app structure:")
    print(mobile_app_structure)

    # Create directories
    import os
    dirs = ['ios', 'android', 'models', 'shared']
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        with open(f"{dir_name}/.gitkeep", 'w') as f:
            f.write("")

    print("Mobile app directories created")
```

### 10.6.2 Core ML for iOS

```python
try:
    import coremltools as ct

    def convert_to_coreml(model, input_shape=(1, 3, 224, 224)):
        """Convert PyTorch model to Core ML for iOS"""

        # Create dummy input
        dummy_input = torch.randn(input_shape)

        # Trace model
        traced_model = torch.jit.trace(model, dummy_input)

        # Convert to Core ML
        mlmodel = ct.convert(
            traced_model,
            inputs=[ct.ImageType(name="input", shape=input_shape)],
            classifier_config=ct.ClassifierConfig(class_labels=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])
        )

        # Save Core ML model
        mlmodel.save("models/cnn_model.mlmodel")

        print("Model converted to Core ML format")

        return mlmodel

except ImportError:
    print("Core ML tools not available. Install with: pip install coremltools")

    def convert_to_coreml(*args, **kwargs):
        raise NotImplementedError("Core ML tools not available")
```

## 10.7 Edge Device Benchmarks

### 10.7.1 Performance Comparison

```python
import time
import psutil
import GPUtil

def benchmark_model(model, test_loader, model_name="Model", num_runs=50):
    """Comprehensive model benchmarking"""

    device = next(model.parameters()).device
    model.eval()

    print(f"\\nBenchmarking {model_name}")
    print("=" * 40)

    # Memory usage before inference
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated()
    else:
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    latencies = []
    cpu_usages = []
    memory_usages = []

    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            if i >= num_runs:
                break

            images = images.to(device)

            # Measure latency
            start_time = time.time()
            outputs = model(images)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            end_time = time.time()

            latency = end_time - start_time
            latencies.append(latency)

            # CPU usage
            cpu_usage = psutil.cpu_percent()
            cpu_usages.append(cpu_usage)

            # Memory usage
            if device.type == 'cuda':
                memory_usage = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            else:
                memory_usage = process.memory_info().rss / 1024 / 1024  # MB
            memory_usages.append(memory_usage)

    # Calculate statistics
    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    p95_latency = sorted(latencies)[int(0.95 * len(latencies))]

    avg_cpu = sum(cpu_usages) / len(cpu_usages)
    peak_memory = max(memory_usages)

    # Model size
    param_count = sum(p.numel() for p in model.parameters())
    model_size = param_count * 4 / (1024 * 1024)  # MB (assuming float32)

    print(f"Model size: {model_size:.2f} MB")
    print(f"Parameters: {param_count:,}")
    print(f"Average latency: {avg_latency:.4f} seconds")
    print(f"Min latency: {min_latency:.4f} seconds")
    print(f"Max latency: {max_latency:.4f} seconds")
    print(f"P95 latency: {p95_latency:.4f} seconds")
    print(f"Average CPU usage: {avg_cpu:.1f}%")
    print(f"Peak memory usage: {peak_memory:.2f} MB")

    return {
        'model_name': model_name,
        'model_size_mb': model_size,
        'param_count': param_count,
        'avg_latency': avg_latency,
        'p95_latency': p95_latency,
        'peak_memory_mb': peak_memory,
        'avg_cpu_percent': avg_cpu
    }

def compare_edge_architectures():
    """Compare different architectures for edge deployment"""

    from chapter8_modern_architectures_code import EfficientNet, MobileNetV3, VisionTransformer

    # Create models
    models = {
        'EfficientNet': EfficientNet(num_classes=10),
        'MobileNetV3': MobileNetV3(num_classes=10),
        'ViT-Tiny': VisionTransformer(
            image_size=224, patch_size=16, num_classes=10,
            embed_dim=192, depth=12, num_heads=3  # Smaller ViT
        )
    }

    # Create test data
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    results = []

    for name, model in models.items():
        try:
            result = benchmark_model(model, test_loader, name, num_runs=20)
            results.append(result)
        except Exception as e:
            print(f"Error benchmarking {name}: {e}")

    # Print comparison table
    print("\\n" + "=" * 80)
    print("EDGE DEPLOYMENT ARCHITECTURE COMPARISON")
    print("=" * 80)
    print("<15")
    print("-" * 80)

    for result in results:
        print("<15")

    return results
```

## 10.8 Summary

This chapter covered essential techniques for deploying CNNs on edge devices:

1. **Model Quantization** - Reducing precision for smaller models
2. **Model Pruning** - Removing unnecessary parameters
3. **Knowledge Distillation** - Transferring knowledge to smaller models
4. **ONNX Export** - Cross-platform model format
5. **TensorRT Optimization** - NVIDIA GPU acceleration
6. **Mobile Deployment** - iOS/Android app integration
7. **Performance Benchmarking** - Measuring edge deployment metrics

Key takeaways:
- Edge deployment requires balancing model accuracy with resource constraints
- Quantization can reduce model size by 4x with minimal accuracy loss
- Pruning and distillation are effective for creating smaller models
- ONNX enables deployment across different platforms and frameworks
- Mobile-specific optimizations are crucial for good user experience

The next chapter will explore advanced computer vision topics and research frontiers.