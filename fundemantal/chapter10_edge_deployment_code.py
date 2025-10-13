import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import time
import os
from datetime import datetime
import json

# Try to import optional dependencies
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("ONNX Runtime not available. Install with: pip install onnxruntime")

try:
    import onnx
    ONNX_AVAILABLE = ONNX_AVAILABLE and True
except ImportError:
    ONNX_AVAILABLE = False
    print("ONNX not available. Install with: pip install onnx")

print("Chapter 10: Edge Deployment and Model Optimization")
print("=" * 60)

# Device configuration
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# 10.1 Model Quantization
class QuantizableEfficientNet(nn.Module):
    """EfficientNet with quantization support"""

    def __init__(self, num_classes=10):
        super(QuantizableEfficientNet, self).__init__()

        # Quantization stubs
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

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
    """Post-training quantization - simplified version"""

    # Set model to evaluation mode
    model.eval()

    try:
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
        print("Model quantized successfully")
        return model

    except Exception as e:
        print(f"Quantization failed, using original model: {e}")
        # Return original model if quantization fails
        return model

def compare_model_sizes(original_model, quantized_model):
    """Compare original vs quantized model sizes"""

    os.makedirs('models', exist_ok=True)

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

# 10.2 Model Pruning
import torch.nn.utils.prune as prune

def apply_weight_pruning(model, pruning_rate=0.3):
    """Apply weight pruning to convolutional layers"""

    print(f"Applying {pruning_rate*100}% weight pruning...")

    # Prune convolutional layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=pruning_rate)
            # Calculate sparsity manually
            total_params = module.weight.numel()
            zero_params = (module.weight == 0).sum().item()
            sparsity = zero_params / total_params
            print(f"Pruned {name}: {sparsity:.1%} sparsity")

    return model

def remove_pruning_masks(model):
    """Remove pruning masks and make pruning permanent"""

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.remove(module, 'weight')

    print("Pruning masks removed - pruning is now permanent")
    return model

# 10.3 Knowledge Distillation
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
                     temperature=3.0, alpha=0.7, epochs=5):
    """Knowledge distillation training"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    teacher_model = teacher_model.to(device).eval()  # Teacher in eval mode
    student_model = student_model.to(device).train()

    distillation_loss = DistillationLoss(temperature, alpha)
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-3, weight_decay=1e-4)

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

# 10.4 ONNX Export and Inference
def export_to_onnx(model, input_shape=(1, 3, 224, 224), onnx_path='models/model.onnx'):
    """Export PyTorch model to ONNX format"""

    if not ONNX_AVAILABLE:
        print("ONNX not available, skipping export")
        return None

    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

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

    if not ONNX_AVAILABLE:
        print("ONNX Runtime not available")
        return None

    # Create ONNX Runtime session
    ort_session = ort.InferenceSession(onnx_path)

    # Prepare input
    ort_inputs = {ort_session.get_inputs()[0].name: input_tensor.numpy()}

    # Run inference
    ort_outputs = ort_session.run(None, ort_inputs)

    return torch.from_numpy(ort_outputs[0])

def compare_pytorch_onnx_inference(pytorch_model, onnx_path, test_loader):
    """Compare PyTorch and ONNX model outputs"""

    if not ONNX_AVAILABLE:
        print("ONNX not available for comparison")
        return

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

            if onnx_output is None:
                return

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

# 10.5 Mobile Optimization
def optimize_for_mobile(model):
    """Optimize model for mobile deployment"""

    # Set to evaluation mode
    model.eval()

    # Fuse operations if available
    if hasattr(model, 'fuse_model'):
        model.fuse_model()

    # Convert to TorchScript
    scripted_model = torch.jit.script(model)

    os.makedirs('models', exist_ok=True)
    scripted_model.save("models/mobile_model.pt")

    print("Model optimized and saved for mobile deployment")

    return scripted_model

# 10.6 Performance Benchmarking
def benchmark_model(model, test_loader, model_name="Model", num_runs=20):
    """Comprehensive model benchmarking"""

    device = next(model.parameters()).device
    model.eval()

    print(f"\\nBenchmarking {model_name}")
    print("=" * 40)

    latencies = []

    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            if i >= num_runs:
                break

            images = images.to(device)

            # Measure latency
            start_time = time.time()
            _ = model(images)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()

            latency = end_time - start_time
            latencies.append(latency)

    # Calculate statistics
    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    p95_latency = sorted(latencies)[int(0.95 * len(latencies))]

    # Model size
    param_count = sum(p.numel() for p in model.parameters())
    model_size = param_count * 4 / (1024 * 1024)  # MB (assuming float32)

    print(f"Model size: {model_size:.2f} MB")
    print(f"Parameters: {param_count:,}")
    print(f"Average latency: {avg_latency:.4f} seconds")
    print(f"Min latency: {min_latency:.4f} seconds")
    print(f"Max latency: {max_latency:.4f} seconds")
    print(f"P95 latency: {p95_latency:.4f} seconds")

    return {
        'model_name': model_name,
        'model_size_mb': model_size,
        'param_count': param_count,
        'avg_latency': avg_latency,
        'p95_latency': p95_latency
    }

# 10.7 Demo and Comparison
def create_test_data():
    """Create test data for demonstrations"""
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Create calibration loader for quantization
    calibration_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    calibration_loader = DataLoader(calibration_dataset, batch_size=32, shuffle=False, num_workers=0)

    return test_loader, calibration_loader

def demo_quantization():
    """Demonstrate model quantization"""

    print("\\n1. Model Quantization Demo")
    print("-" * 30)

    # Create model
    model = QuantizableEfficientNet(num_classes=10)
    test_loader, calibration_loader = create_test_data()

    print(f"Original model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Benchmark original model
    benchmark_model(model, test_loader, "Original Model", num_runs=5)

    # Quantize model
    try:
        quantized_model = quantize_model(model, calibration_loader)

        # Compare sizes
        _, _ = compare_model_sizes(model, quantized_model)

        # Benchmark quantized model
        benchmark_model(quantized_model, test_loader, "Quantized Model", num_runs=5)

        print("\\nQuantization Results:")
        print(".1f")
        print(".3f")

    except Exception as e:
        print(f"Quantization failed: {e}")

def demo_pruning():
    """Demonstrate model pruning"""

    print("\\n2. Model Pruning Demo")
    print("-" * 30)

    # Create model
    model = QuantizableEfficientNet(num_classes=10)
    test_loader, _ = create_test_data()

    print(f"Original model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Benchmark original model
    benchmark_model(model, test_loader, "Original Model", num_runs=5)

    # Apply pruning
    pruned_model = apply_weight_pruning(model, pruning_rate=0.3)

    # Remove pruning masks
    pruned_model = remove_pruning_masks(pruned_model)

    print(f"Pruned model parameters: {sum(p.numel() for p in pruned_model.parameters()):,}")

    # Benchmark pruned model
    benchmark_model(pruned_model, test_loader, "Pruned Model", num_runs=5)

def demo_knowledge_distillation():
    """Demonstrate knowledge distillation"""

    print("\\n3. Knowledge Distillation Demo")
    print("-" * 30)

    # Create teacher and student models
    teacher_model = QuantizableEfficientNet(num_classes=10)  # Larger teacher
    student_model = nn.Sequential(  # Simple student
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, 10)
    )

    train_loader, test_loader = create_test_data()

    print(f"Teacher parameters: {sum(p.numel() for p in teacher_model.parameters()):,}")
    print(f"Student parameters: {sum(p.numel() for p in student_model.parameters()):,}")

    # Benchmark original student
    benchmark_model(student_model, test_loader, "Original Student", num_runs=5)

    # Train student with distillation
    distilled_student = distill_knowledge(
        teacher_model, student_model, train_loader, test_loader,
        temperature=3.0, alpha=0.7, epochs=2
    )

    # Benchmark distilled student
    benchmark_model(distilled_student, test_loader, "Distilled Student", num_runs=5)

def demo_onnx_export():
    """Demonstrate ONNX export and inference"""

    print("\\n4. ONNX Export Demo")
    print("-" * 30)

    if not ONNX_AVAILABLE:
        print("ONNX not available, skipping demo")
        return

    # Create model
    model = QuantizableEfficientNet(num_classes=10)
    test_loader, _ = create_test_data()

    # Export to ONNX
    onnx_path = export_to_onnx(model, onnx_path='models/demo_model.onnx')

    if onnx_path:
        # Compare PyTorch vs ONNX inference
        compare_pytorch_onnx_inference(model, onnx_path, test_loader)

def demo_mobile_optimization():
    """Demonstrate mobile optimization"""

    print("\\n5. Mobile Optimization Demo")
    print("-" * 30)

    # Create model
    model = QuantizableEfficientNet(num_classes=10)

    # Optimize for mobile
    optimize_for_mobile(model)

    print("Mobile optimization completed")
    print("Model saved as TorchScript for mobile deployment")

def run_edge_deployment_demo():
    """Run complete edge deployment demonstration"""

    print("Edge Deployment and Model Optimization Demo")
    print("=" * 50)

    try:
        demo_quantization()
        demo_pruning()
        demo_knowledge_distillation()
        demo_onnx_export()
        demo_mobile_optimization()

        print("\\n" + "=" * 50)
        print("Edge Deployment Demo Complete!")
        print("=" * 50)
        print("Summary of Techniques:")
        print("• Quantization: Reduces model size by ~4x")
        print("• Pruning: Removes unnecessary parameters")
        print("• Distillation: Transfers knowledge to smaller models")
        print("• ONNX: Enables cross-platform deployment")
        print("• Mobile Optimization: TorchScript for edge devices")

    except Exception as e:
        print(f"Demo failed: {e}")

if __name__ == "__main__":
    run_edge_deployment_demo()