# Chapter 9: MLOps and Model Deployment

## 9.1 Introduction to MLOps

MLOps (Machine Learning Operations) is the practice of applying DevOps principles to machine learning workflows. It encompasses the entire ML lifecycle from development to deployment and monitoring.

### Key MLOps Components:
1. **Experiment Tracking** - Log parameters, metrics, and artifacts
2. **Model Versioning** - Track model versions and lineage
3. **Model Registry** - Store and manage trained models
4. **CI/CD Pipelines** - Automated testing and deployment
5. **Model Monitoring** - Track performance in production
6. **Model Serving** - Deploy models for inference

## 9.2 MLflow for Experiment Tracking

MLflow is an open-source platform for managing the ML lifecycle.

### 9.2.1 Setting Up MLflow

```python
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import time

# Set MLflow tracking URI (local server)
mlflow.set_tracking_uri("http://localhost:5000")

# Create or get experiment
experiment_name = "CNN_Architectures_Comparison"
mlflow.set_experiment(experiment_name)
```

### 9.2.2 Logging Experiments with MLflow

```python
def train_with_mlflow(model, model_name, train_loader, test_loader, epochs=5):
    """Train model with MLflow tracking"""

    with mlflow.start_run(run_name=f"{model_name}_training"):
        # Log model architecture info
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", 32)
        mlflow.log_param("num_parameters", sum(p.numel() for p in model.parameters()))

        # Training setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        best_accuracy = 0.0

        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            train_accuracy = 100. * correct / total
            avg_train_loss = train_loss / len(train_loader)

            # Validation phase
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

            val_accuracy = 100. * correct / total
            avg_val_loss = val_loss / len(test_loader)

            # Log metrics
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_accuracy, step=epoch)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)

            print(f"Epoch {epoch+1}/{epochs}: Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%")

            # Save best model
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                mlflow.pytorch.log_model(model, "best_model")

        # Log final metrics
        mlflow.log_metric("best_accuracy", best_accuracy)

        return best_accuracy
```

## 9.3 Model Versioning and Registry

### 9.3.1 Model Serialization

```python
def save_model_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """Save model checkpoint with metadata"""

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
        'timestamp': time.time(),
        'model_architecture': str(model.__class__.__name__)
    }

    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")

def load_model_checkpoint(filepath, model, optimizer=None):
    """Load model checkpoint"""

    checkpoint = torch.load(filepath)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    accuracy = checkpoint['accuracy']

    print(f"Checkpoint loaded from {filepath}")
    print(f"Epoch: {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")

    return epoch, loss, accuracy
```

### 9.3.2 Model Registry with MLflow

```python
def register_model_with_mlflow(model, model_name, accuracy):
    """Register model in MLflow Model Registry"""

    # Log the model
    with mlflow.start_run(run_name=f"{model_name}_registration"):
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("accuracy", accuracy)

        # Log model
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name=model_name
        )

    # Transition to production if accuracy > threshold
    client = mlflow.tracking.MlflowClient()

    # Get latest version
    latest_version = client.get_latest_versions(model_name, stages=["None"])[0]

    if accuracy > 85.0:  # Production threshold
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version.version,
            stage="Production"
        )
        print(f"Model {model_name} v{latest_version.version} promoted to Production")
    else:
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version.version,
            stage="Staging"
        )
        print(f"Model {model_name} v{latest_version.version} moved to Staging")
```

## 9.4 CI/CD Pipeline for ML Models

### 9.4.1 GitHub Actions Workflow

Create `.github/workflows/ml-pipeline.yml`:

```yaml
name: ML Training and Deployment Pipeline

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt

    - name: Run tests
      run: |
        python -m pytest tests/ -v

    - name: Run linting
      run: |
        flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 src/ --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

  train:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install mlflow

    - name: Train model
      run: |
        python src/train.py --experiment-name ${{ github.run_number }}

    - name: Upload model artifacts
      uses: actions/upload-artifact@v3
      with:
        name: trained-model
        path: models/

  deploy:
    needs: train
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
    - uses: actions/checkout@v3

    - name: Download model artifacts
      uses: actions/download-artifact@v3
      with:
        name: trained-model
        path: models/

    - name: Deploy to staging
      run: |
        echo "Deploying model to staging environment"
        # Add your deployment commands here
```

### 9.4.2 Model Validation Tests

```python
import pytest
import torch
import numpy as np
from src.models import EfficientNet, MobileNetV3, VisionTransformer

class TestModelValidation:
    """Test model validation and performance"""

    def setup_method(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.test_input = torch.randn(1, 3, 224, 224).to(self.device)

    def test_model_forward_pass(self):
        """Test that models can perform forward pass"""
        models = [
            EfficientNet(num_classes=10),
            MobileNetV3(num_classes=10),
            VisionTransformer(num_classes=10)
        ]

        for model in models:
            model = model.to(self.device)
            model.eval()

            with torch.no_grad():
                output = model(self.test_input)
                assert output.shape == (1, 10), f"Wrong output shape for {model.__class__.__name__}"

    def test_model_parameter_count(self):
        """Test model parameter counts are reasonable"""
        efficientnet = EfficientNet(num_classes=10)
        mobilenet = MobileNetV3(num_classes=10)
        vit = VisionTransformer(num_classes=10)

        eff_params = sum(p.numel() for p in efficientnet.parameters())
        mob_params = sum(p.numel() for p in mobilenet.parameters())
        vit_params = sum(p.numel() for p in vit.parameters())

        # EfficientNet should have reasonable parameter count
        assert 5_000_000 < eff_params < 10_000_000

        # MobileNet should be more efficient
        assert 2_000_000 < mob_params < 6_000_000

        # ViT should have more parameters
        assert 8_000_000 < vit_params < 15_000_000

    def test_model_accuracy_threshold(self):
        """Test that trained models meet minimum accuracy"""
        # This would load a pre-trained model and test on validation set
        # For now, just check the testing framework works
        assert True  # Placeholder

    def test_model_inference_time(self):
        """Test model inference time is reasonable"""
        model = EfficientNet(num_classes=10).to(self.device)
        model.eval()

        # Warm up
        with torch.no_grad():
            for _ in range(10):
                _ = model(self.test_input)

        # Time inference
        import time
        start_time = time.time()

        with torch.no_grad():
            for _ in range(100):
                _ = model(self.test_input)

        end_time = time.time()
        avg_time = (end_time - start_time) / 100

        # Should be reasonably fast (< 50ms on GPU, < 200ms on CPU)
        max_time = 0.2 if str(self.device) == 'cpu' else 0.05
        assert avg_time < max_time, f"Inference too slow: {avg_time:.4f}s"
```

## 9.5 Model Monitoring and Drift Detection

### 9.5.1 Performance Monitoring

```python
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json

class ModelMonitor:
    """Monitor model performance in production"""

    def __init__(self, model_name, threshold=0.85):
        self.model_name = model_name
        self.threshold = threshold
        self.metrics_history = []
        self.alerts = []

    def log_prediction(self, prediction, actual=None, confidence=None):
        """Log a prediction for monitoring"""

        entry = {
            'timestamp': datetime.now(),
            'prediction': prediction,
            'actual': actual,
            'confidence': confidence,
            'correct': prediction == actual if actual is not None else None
        }

        self.metrics_history.append(entry)

        # Keep only last 1000 predictions
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]

    def calculate_metrics(self):
        """Calculate current performance metrics"""

        if not self.metrics_history:
            return {}

        recent_predictions = self.metrics_history[-100:]  # Last 100 predictions

        if all(entry['actual'] is None for entry in recent_predictions):
            return {}

        correct_predictions = [entry for entry in recent_predictions
                             if entry['correct'] is True]

        accuracy = len(correct_predictions) / len(recent_predictions)

        # Calculate confidence statistics
        confidences = [entry['confidence'] for entry in recent_predictions
                      if entry['confidence'] is not None]

        metrics = {
            'accuracy': accuracy,
            'avg_confidence': np.mean(confidences) if confidences else None,
            'min_confidence': np.min(confidences) if confidences else None,
            'max_confidence': np.max(confidences) if confidences else None,
            'timestamp': datetime.now()
        }

        return metrics

    def check_for_alerts(self):
        """Check if model performance has degraded"""

        metrics = self.calculate_metrics()

        if not metrics or 'accuracy' not in metrics:
            return []

        current_accuracy = metrics['accuracy']

        if current_accuracy < self.threshold:
            alert = {
                'type': 'accuracy_drop',
                'message': f"Model accuracy dropped to {current_accuracy:.3f} (threshold: {self.threshold})",
                'timestamp': datetime.now(),
                'metrics': metrics
            }
            self.alerts.append(alert)
            return [alert]

        return []

    def get_performance_report(self):
        """Generate performance report"""

        metrics = self.calculate_metrics()

        report = {
            'model_name': self.model_name,
            'metrics': metrics,
            'alerts': self.alerts[-10:],  # Last 10 alerts
            'total_predictions': len(self.metrics_history),
            'report_generated': datetime.now()
        }

        return report
```

### 9.5.2 Data Drift Detection

```python
from scipy.stats import ks_2samp, chi2_contingency
import numpy as np

class DriftDetector:
    """Detect data drift in production"""

    def __init__(self, reference_data=None):
        self.reference_data = reference_data
        self.drift_threshold = 0.05  # p-value threshold

    def update_reference_data(self, data):
        """Update reference data distribution"""
        self.reference_data = data

    def detect_drift(self, current_data, feature_name="feature"):
        """Detect if current data distribution differs from reference"""

        if self.reference_data is None:
            return {'drift_detected': False, 'p_value': 1.0}

        # Kolmogorov-Smirnov test for continuous data
        if len(current_data.shape) == 1 or current_data.shape[1] == 1:
            stat, p_value = ks_2samp(self.reference_data.flatten(), current_data.flatten())
        else:
            # For multi-dimensional data, test each feature
            p_values = []
            for i in range(current_data.shape[1]):
                stat, p_val = ks_2samp(self.reference_data[:, i], current_data[:, i])
                p_values.append(p_val)

            p_value = np.min(p_values)  # Most significant difference

        drift_detected = p_value < self.drift_threshold

        return {
            'drift_detected': drift_detected,
            'p_value': p_value,
            'feature': feature_name,
            'test_statistic': stat if 'stat' in locals() else None
        }

    def detect_concept_drift(self, predictions, actuals):
        """Detect concept drift by comparing prediction accuracy over time"""

        if len(predictions) < 100:
            return {'drift_detected': False}

        # Split into two windows
        window_size = len(predictions) // 2
        recent_predictions = predictions[-window_size:]
        recent_actuals = actuals[-window_size:]
        older_predictions = predictions[:-window_size]
        older_actuals = actuals[:-window_size]

        # Calculate accuracy for each window
        recent_accuracy = np.mean(recent_predictions == recent_actuals)
        older_accuracy = np.mean(older_predictions == older_actuals)

        # Check for significant drop
        accuracy_drop = older_accuracy - recent_accuracy
        drift_detected = accuracy_drop > 0.1  # 10% drop threshold

        return {
            'drift_detected': drift_detected,
            'recent_accuracy': recent_accuracy,
            'older_accuracy': older_accuracy,
            'accuracy_drop': accuracy_drop
        }
```

## 9.6 Model Serving with FastAPI

### 9.6.1 FastAPI Model Server

```python
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import numpy as np
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CNN Model Serving API", version="1.0.0")

class ModelManager:
    """Manage model loading and inference"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                           'dog', 'frog', 'horse', 'ship', 'truck']

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def load_model(self, model_path: str, model_class):
        """Load model from path"""
        try:
            self.model = model_class(num_classes=10)
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def predict(self, image: Image.Image) -> dict:
        """Make prediction on image"""
        try:
            # Preprocess image
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # Inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)

            # Prepare response
            prediction = {
                'class_id': predicted_class.item(),
                'class_name': self.class_names[predicted_class.item()],
                'confidence': confidence.item(),
                'probabilities': probabilities.squeeze().cpu().numpy().tolist(),
                'timestamp': datetime.now().isoformat()
            }

            return prediction

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

# Global model manager
model_manager = ModelManager()

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    # Load your trained model here
    # model_manager.load_model("models/efficientnet_cifar10.pth", EfficientNet)
    logger.info("Model serving API started")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict image class"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Read image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')

        # Make prediction
        prediction = model_manager.predict(image)

        return JSONResponse(content=prediction)

    except Exception as e:
        logger.error(f"Prediction endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/model_info")
async def model_info():
    """Get model information"""
    if model_manager.model is None:
        return {"status": "no_model_loaded"}

    return {
        "model_loaded": True,
        "device": str(model_manager.device),
        "num_parameters": sum(p.numel() for p in model_manager.model.parameters()),
        "class_names": model_manager.class_names
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 9.6.2 Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY models/ ./models/

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \\
    && chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["python", "src/app.py"]
```

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  ml-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/models/efficientnet_cifar10.pth
    volumes:
      - ./models:/app/models:ro
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  mlflow-server:
    image: ghcr.io/mlflow/mlflow:v2.8.0
    ports:
      - "5000:5000"
    environment:
      - MLFLOW_BACKEND_STORE_URI=sqlite:///mlflow.db
      - MLFLOW_ARTIFACT_STORE_URI=./mlflow-artifacts
    volumes:
      - ./mlflow-data:/mlflow-data
    command: mlflow server --host 0.0.0.0 --port 5000
```

## 9.7 Summary

This chapter covered essential MLOps practices:

1. **Experiment Tracking** with MLflow
2. **Model Versioning** and registry management
3. **CI/CD Pipelines** for automated ML workflows
4. **Model Monitoring** and drift detection
5. **Model Serving** with FastAPI and Docker

Key takeaways:
- MLOps bridges the gap between ML development and production deployment
- Experiment tracking ensures reproducibility and collaboration
- Continuous monitoring prevents model performance degradation
- Containerization ensures consistent deployment across environments

The next chapter will focus on edge deployment and model optimization for resource-constrained devices.