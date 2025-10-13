# Chapter 6: Model Deployment and Inference

## 6.1 Model Saving and Loading

### State Dict (Recommended)
```python
# Save model parameters
torch.save(model.state_dict(), 'model_weights.pth')

# Load model parameters
model = YourModelClass()
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()
```

### Complete Model Saving
```python
# Save entire model (includes architecture)
torch.save(model, 'complete_model.pth')

# Load complete model
model = torch.load('complete_model.pth')
model.eval()
```

### Checkpoint Saving (Training Resume)
```python
# Save training checkpoint
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'loss': best_loss,
    'accuracy': best_acc
}
torch.save(checkpoint, 'checkpoint.pth')

# Load checkpoint
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
start_epoch = checkpoint['epoch']
```

## 6.2 Model Optimization for Inference

### TorchScript Conversion
```python
# Convert to TorchScript for faster inference
model.eval()
example_input = torch.randn(1, 3, 32, 32)

# Method 1: Tracing
traced_model = torch.jit.trace(model, example_input)
torch.jit.save(traced_model, 'model_traced.pt')

# Method 2: Scripting (more flexible)
scripted_model = torch.jit.script(model)
torch.jit.save(scripted_model, 'model_scripted.pt')

# Load and use
loaded_model = torch.jit.load('model_traced.pt')
with torch.no_grad():
    output = loaded_model(example_input)
```

### ONNX Export
```python
# Export to ONNX format (cross-platform)
torch.onnx.export(
    model,
    example_input,
    'model.onnx',
    export_params=True,
    opset_version=11,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
```

### Quantization
```python
# Dynamic quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
torch.save(quantized_model, 'model_quantized.pth')

# Check model size reduction
original_size = os.path.getsize('model_weights.pth')
quantized_size = os.path.getsize('model_quantized.pth')
print(f"Original: {original_size} bytes")
print(f"Quantized: {quantized_size} bytes")
print(f"Reduction: {(1 - quantized_size/original_size)*100:.1f}%")
```

## 6.3 Inference Optimization

### Batch Processing
```python
def batch_inference(model, images, batch_size=32):
    """Process images in batches for efficient inference"""
    model.eval()
    all_predictions = []

    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            batch_tensor = torch.stack(batch).to(device)

            outputs = model(batch_tensor)
            predictions = outputs.argmax(dim=1)

            all_predictions.extend(predictions.cpu().numpy())

    return all_predictions
```

### Memory Optimization
```python
# Use autocast for mixed precision inference
from torch.cuda.amp import autocast

@torch.no_grad()
def efficient_inference(model, input_tensor):
    model.eval()

    with autocast():
        output = model(input_tensor)

    return output
```

### GPU Optimization
```python
# Pin memory for faster GPU transfer
train_loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True  # Faster GPU transfer
)

# Use non-blocking transfers
inputs = inputs.to(device, non_blocking=True)
targets = targets.to(device, non_blocking=True)
```

## 6.4 Production Inference Pipeline

### Image Preprocessing Pipeline
```python
from PIL import Image
import torchvision.transforms as transforms

class ImageProcessor:
    def __init__(self, input_size=(32, 32), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        self.transforms = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def process_image(self, image_path):
        """Process single image"""
        image = Image.open(image_path).convert('RGB')
        return self.transforms(image).unsqueeze(0)

    def process_batch(self, image_paths):
        """Process batch of images"""
        images = []
        for path in image_paths:
            image = Image.open(path).convert('RGB')
            images.append(self.transforms(image))
        return torch.stack(images)

# Usage
processor = ImageProcessor()
input_tensor = processor.process_image('image.jpg')
```

### Inference Class
```python
class ModelInference:
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        self.model = self.load_model(model_path)
        self.processor = ImageProcessor()

    def load_model(self, model_path):
        """Load trained model"""
        model = YourModelClass()  # Replace with your model class
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

    def predict_batch(self, image_paths, batch_size=32):
        """Predict batch of images"""
        predictions = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_tensor = self.processor.process_batch(batch_paths).to(self.device)

            with torch.no_grad():
                outputs = self.model(batch_tensor)
                batch_predictions = outputs.argmax(dim=1).cpu().numpy()
                predictions.extend(batch_predictions)

        return predictions

# Usage
inference = ModelInference('model_weights.pth', device='cuda')
predicted_class, confidence, probs = inference.predict_single('test_image.jpg')
print(f"Predicted class: {predicted_class}, Confidence: {confidence:.2f}")
```

## 6.5 Performance Benchmarking

### Inference Speed Test
```python
import time

def benchmark_inference(model, input_tensor, num_runs=100):
    """Benchmark inference speed"""
    model.eval()
    times = []

    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = model(input_tensor)

        # Benchmark
        for _ in range(num_runs):
            start_time = time.time()
            _ = model(input_tensor)
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # Wait for GPU
            end_time = time.time()
            times.append(end_time - start_time)

    avg_time = sum(times) / len(times)
    fps = 1.0 / avg_time

    print(f"Average inference time: {avg_time*1000:.2f} ms")
    print(f"FPS: {fps:.2f}")
    print(f"Min time: {min(times)*1000:.2f} ms")
    print(f"Max time: {max(times)*1000:.2f} ms")

    return avg_time, fps

# Benchmark
example_input = torch.randn(1, 3, 32, 32).to(device)
avg_time, fps = benchmark_inference(model, example_input)
```

### Memory Usage
```python
def check_memory_usage():
    """Check GPU/CPU memory usage"""
    if torch.cuda.is_available():
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
        print(f"GPU Memory Cached: {torch.cuda.memory_reserved()/1024**2:.1f} MB")
    else:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        print(f"CPU Memory Usage: {memory_info.rss/1024**2:.1f} MB")

check_memory_usage()
```

## 6.6 Web API Deployment (Flask)

### Simple Flask API
```python
from flask import Flask, request, jsonify
import io
import base64
from PIL import Image

app = Flask(__name__)

# Load model
inference = ModelInference('model_weights.pth')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for image prediction"""
    try:
        # Get image from request
        image_data = request.json['image']  # Base64 encoded image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Save temporarily and predict
        temp_path = '/tmp/temp_image.jpg'
        image.save(temp_path)

        predicted_class, confidence, probabilities = inference.predict_single(temp_path)

        # Clean up
        os.remove(temp_path)

        # Return results
        result = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities.tolist()
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### FastAPI (Modern Alternative)
```python
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import shutil
import os

app = FastAPI()

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """FastAPI endpoint for image prediction"""
    try:
        # Save uploaded file temporarily
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Make prediction
        predicted_class, confidence, probabilities = inference.predict_single(temp_path)

        # Clean up
        os.remove(temp_path)

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities.tolist()
        }

    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

# Run with: uvicorn main:app --reload
```

## 6.7 Docker Deployment

### Dockerfile
```dockerfile
FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and application code
COPY model_weights.pth .
COPY app.py .

# Expose port
EXPOSE 5000

# Run application
CMD ["python", "app.py"]
```

### requirements.txt
```
torch==1.9.0
torchvision==0.10.0
flask==2.0.1
pillow==8.3.1
numpy==1.21.2
```

### Build and Run
```bash
# Build Docker image
docker build -t cnn-inference .

# Run container
docker run -p 5000:5000 cnn-inference
```

## 6.8 Cloud Deployment

### AWS Lambda
```python
import json
import base64
import torch
from PIL import Image
import io

def lambda_handler(event, context):
    """AWS Lambda handler for inference"""

    # Decode base64 image
    image_data = base64.b64decode(event['image'])
    image = Image.open(io.BytesIO(image_data)).convert('RGB')

    # Process and predict
    processor = ImageProcessor()
    input_tensor = processor.process_image_from_pil(image).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = output.argmax().item()

    return {
        'statusCode': 200,
        'body': json.dumps({
            'predicted_class': predicted_class
        })
    }
```

### Google Cloud Functions
```python
def predict_image(request):
    """Google Cloud Function"""
    request_json = request.get_json()

    if not request_json or 'image' not in request_json:
        return ('Missing image data', 400)

    # Decode and process image
    image_data = base64.b64decode(request_json['image'])
    image = Image.open(io.BytesIO(image_data)).convert('RGB')

    # Make prediction
    predicted_class, confidence, _ = inference.predict_single_from_pil(image)

    return {
        'predicted_class': predicted_class,
        'confidence': confidence
    }
```

## 6.9 Monitoring and Logging

### Basic Logging
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('inference.log'),
        logging.StreamHandler()
    ]
)

def predict_with_logging(image_path):
    """Prediction with logging"""
    start_time = time.time()

    try:
        result = inference.predict_single(image_path)
        inference_time = time.time() - start_time

        logging.info(f"Prediction successful - Class: {result[0]}, Confidence: {result[1]:.2f}, Time: {inference_time:.3f}s")
        return result

    except Exception as e:
        logging.error(f"Prediction failed for {image_path}: {str(e)}")
        raise
```

### Performance Monitoring
```python
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

# Use in prediction function
def monitored_predict(image_path):
    start_time = time.time()
    try:
        result = inference.predict_single(image_path)
        inference_time = time.time() - start_time
        monitor.log_request(inference_time, success=True)
        return result
    except Exception as e:
        inference_time = time.time() - start_time
        monitor.log_request(inference_time, success=False)
        raise
```

---

**Next:** Chapter 7 - Best Practices and Troubleshooting (Optimization Tips, Common Issues, Advanced Topics)!