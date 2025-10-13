import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
import time
import os
import logging

print("Chapter 6: Model Deployment and Inference")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Device configuration
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# Define a simple model for demonstration
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

# 6.1 Model Saving and Loading
print("\n6.1 Model Saving and Loading")

model = SimpleCNN()

# Save state dict (recommended)
torch.save(model.state_dict(), 'model_weights.pth')
print("Model weights saved as 'model_weights.pth'")

# Load state dict
loaded_model = SimpleCNN()
loaded_model.load_state_dict(torch.load('model_weights.pth'))
loaded_model.to(device)
loaded_model.eval()
print("Model loaded from state dict")

# Save complete model
torch.save(model, 'complete_model.pth')
print("Complete model saved as 'complete_model.pth'")

# Load complete model
complete_model = torch.load('complete_model.pth', weights_only=False)
complete_model.to(device)
complete_model.eval()
print("Complete model loaded")

# 6.2 Model Optimization
print("\n6.2 Model Optimization")

# TorchScript conversion
model.eval()
example_input = torch.randn(1, 3, 32, 32).cpu()  # Use CPU tensor for TorchScript

# Tracing
traced_model = torch.jit.trace(model, example_input)
torch.jit.save(traced_model, 'model_traced.pt')
print("TorchScript traced model saved")

# Scripting
try:
    scripted_model = torch.jit.script(model)
    torch.jit.save(scripted_model, 'model_scripted.pt')
    print("TorchScript scripted model saved")
except Exception as e:
    print(f"Scripting failed: {e}")

# Load traced model
jit_model = torch.jit.load('model_traced.pt')
print("TorchScript model loaded")

# 6.3 Inference Pipeline
print("\n6.3 Inference Pipeline")

class ImageProcessor:
    def __init__(self, input_size=(32, 32), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        self.transforms = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def process_image(self, image_path):
        """Process single image from file"""
        image = Image.open(image_path).convert('RGB')
        return self.transforms(image).unsqueeze(0)

    def process_image_from_pil(self, pil_image):
        """Process PIL image directly"""
        return self.transforms(pil_image).unsqueeze(0)

class ModelInference:
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        self.model = self.load_model(model_path)
        self.processor = ImageProcessor()

    def load_model(self, model_path):
        """Load trained model"""
        if model_path.endswith('.pt'):
            # TorchScript model
            model = torch.jit.load(model_path)
        else:
            # Regular PyTorch model
            model = SimpleCNN()
            model.load_state_dict(torch.load(model_path, map_location=self.device))

        model.to(self.device)
        model.eval()
        return model

    def predict_single(self, image_path):
        """Predict single image"""
        input_tensor = self.processor.process_image(image_path).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted_class = probabilities.max(1)

        return predicted_class.item(), confidence.item(), probabilities.squeeze().cpu().numpy()

    def predict_single_from_pil(self, pil_image):
        """Predict from PIL image"""
        input_tensor = self.processor.process_image_from_pil(pil_image).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted_class = probabilities.max(1)

        return predicted_class.item(), confidence.item(), probabilities.squeeze().cpu().numpy()

# Initialize inference
inference = ModelInference('model_weights.pth', device=device.type)

# 6.4 Performance Benchmarking
print("\n6.4 Performance Benchmarking")

def benchmark_inference(model, input_tensor, num_runs=50):
    """Benchmark inference speed"""
    model.eval()
    times = []

    with torch.no_grad():
        # Warmup
        for _ in range(5):
            _ = model(input_tensor)

        # Benchmark
        for _ in range(num_runs):
            start_time = time.time()
            _ = model(input_tensor)
            if device.type == 'cuda':
                torch.mps.synchronize()  # Wait for GPU
            end_time = time.time()
            times.append(end_time - start_time)

    avg_time = sum(times) / len(times)
    fps = 1.0 / avg_time

    print(f"Average inference time: {avg_time*1000:.2f} ms")
    print(f"FPS: {fps:.2f}")
    print(f"Min time: {min(times)*1000:.2f} ms")
    print(f"Max time: {max(times)*1000:.2f} ms")

    return avg_time, fps

# Benchmark different models
example_input = torch.randn(1, 3, 32, 32).to(device)

print("\nBenchmarking regular model:")
benchmark_inference(loaded_model, example_input.to(device))

print("\nBenchmarking TorchScript model:")
benchmark_inference(jit_model, example_input.cpu())  # TorchScript needs CPU tensors

# 6.5 Batch Processing
print("\n6.5 Batch Processing")

def batch_inference(model, image_tensors, batch_size=32):
    """Process images in batches"""
    model.eval()
    all_predictions = []

    with torch.no_grad():
        for i in range(0, len(image_tensors), batch_size):
            batch = image_tensors[i:i+batch_size]
            batch_tensor = torch.stack(batch).to(device)

            outputs = model(batch_tensor)
            predictions = outputs.argmax(dim=1)

            all_predictions.extend(predictions.cpu().numpy())

    return all_predictions

# Create batch of dummy images
batch_images = [torch.randn(3, 32, 32) for _ in range(10)]
batch_predictions = batch_inference(loaded_model, batch_images, batch_size=5)
print(f"Batch predictions: {batch_predictions}")

# 6.6 Memory Usage
print("\n6.6 Memory Usage")

def check_memory_usage():
    """Check memory usage"""
    if device.type == 'cuda':
        print(f"GPU Memory Allocated: {torch.mps.current_allocated_memory()/1024**2:.1f} MB")
        print(f"GPU Memory Cached: {torch.mps.driver_allocated_memory()/1024**2:.1f} MB")
    else:
        print("CPU memory monitoring not implemented in this demo")

check_memory_usage()

# 6.7 Model Size Comparison
print("\n6.7 Model Size Comparison")

def get_model_size(model_path):
    """Get file size in MB"""
    size_bytes = os.path.getsize(model_path)
    size_mb = size_bytes / (1024 * 1024)
    return size_mb

model_files = ['model_weights.pth', 'complete_model.pth', 'model_traced.pt']
for file in model_files:
    if os.path.exists(file):
        size = get_model_size(file)
        print(f"{file}: {size:.2f} MB")

# 6.8 Logging Setup
print("\n6.8 Logging Setup")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('inference.log'),
        logging.StreamHandler()
    ]
)

class InferenceMonitor:
    def __init__(self):
        self.total_requests = 0
        self.total_time = 0
        self.errors = 0

    def log_request(self, inference_time, success=True):
        self.total_requests += 1
        self.total_time += inference_time
        if not success:
            self.errors += 1

    def get_stats(self):
        avg_time = self.total_time / self.total_requests if self.total_requests > 0 else 0
        error_rate = self.errors / self.total_requests if self.total_requests > 0 else 0

        return {
            'total_requests': self.total_requests,
            'average_time': avg_time,
            'error_rate': error_rate
        }

monitor = InferenceMonitor()

def monitored_predict(image_path):
    """Prediction with monitoring"""
    start_time = time.time()
    try:
        result = inference.predict_single(image_path)
        inference_time = time.time() - start_time
        monitor.log_request(inference_time, success=True)
        logging.info(f"Prediction successful - Class: {result[0]}, Confidence: {result[1]:.2f}, Time: {inference_time:.3f}s")
        return result
    except Exception as e:
        inference_time = time.time() - start_time
        monitor.log_request(inference_time, success=False)
        logging.error(f"Prediction failed for {image_path}: {str(e)}")
        raise

# 6.9 Web API Example (Flask-like)
print("\n6.9 Web API Example")

class MockFlaskApp:
    """Mock Flask app for demonstration"""
    def __init__(self):
        self.routes = {}

    def route(self, path, methods=None):
        def decorator(func):
            self.routes[path] = func
            return func
        return decorator

    def run_request(self, path, data):
        """Simulate API request"""
        if path in self.routes:
            # Mock request object
            class MockRequest:
                def __init__(self, json_data):
                    self.json = json_data

            request = MockRequest(data)
            return self.routes[path]()
        else:
            return {"error": "Route not found"}, 404

# Create mock API
app = MockFlaskApp()

@app.route('/predict', methods=['POST'])
def predict_api():
    """Mock API endpoint"""
    try:
        # In real Flask, this would be: image_data = request.json['image']
        # For demo, we'll use a dummy base64 string
        dummy_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="

        # Decode base64 (dummy)
        image_data = base64.b64decode(dummy_b64)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')

        # Make prediction
        predicted_class, confidence, probabilities = inference.predict_single_from_pil(image)

        result = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities.tolist()
        }

        print(f"API Prediction: Class {predicted_class}, Confidence {confidence:.2f}")
        return result

    except Exception as e:
        print(f"API Error: {str(e)}")
        return {'error': str(e)}, 400

# Test the mock API
print("Testing mock API...")
response = app.run_request('/predict', {'image': 'dummy'})
print(f"API Response: {response}")

# 6.10 Final Statistics
print("\n6.10 Final Statistics")

stats = monitor.get_stats()
print(f"Total requests: {stats['total_requests']}")
print(f"Average inference time: {stats['average_time']:.3f}s")
print(f"Error rate: {stats['error_rate']:.2%}")

print("\n" + "="*60)
print("Chapter 6 Summary:")
print("- Models saved and loaded successfully")
print("- TorchScript conversion demonstrated")
print("- Inference pipeline created")
print("- Performance benchmarking completed")
print("- Memory usage monitored")
print("- Mock API implemented")
print("- Logging and monitoring set up")
print("\nDeployment ready! Next: Chapter 7 - Best Practices!")