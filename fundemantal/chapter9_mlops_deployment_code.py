import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
# Try to import MLflow (optional dependency)
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    print("Warning: MLflow not available. Install with: pip install mlflow")
    MLFLOW_AVAILABLE = False
    # Create dummy classes for compatibility when MLflow is not available
    class mlflow:
        @staticmethod
        def set_experiment(name):
            """Dummy method when MLflow is not available"""
            pass

        @staticmethod
        def start_run(run_name=None):
            """Dummy method when MLflow is not available"""
            return DummyRun()

        @staticmethod
        def log_param(key, value):
            """Dummy method when MLflow is not available"""
            pass

        @staticmethod
        def log_metric(key, value, step=None):
            """Dummy method when MLflow is not available"""
            pass

        class pytorch:
            @staticmethod
            def log_model(model, name):
                """Dummy method when MLflow is not available"""
                pass

    class DummyRun:
        def __enter__(self):
            """Dummy context manager entry"""
            return self

        def __exit__(self, *args):
            """Dummy context manager exit"""
            pass
import numpy as np
import time
import os
from datetime import datetime
import json
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import uvicorn
from typing import Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("Chapter 9: MLOps and Model Deployment")
print("=" * 50)

# 9.1 MLflow Experiment Tracking
class MLflowExperimentTracker:
    """MLflow experiment tracking wrapper"""

    def __init__(self, experiment_name="CNN_Training"):
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)

    def log_params(self, params_dict):
        """Log parameters"""
        for key, value in params_dict.items():
            mlflow.log_param(key, value)

    def log_metrics(self, metrics_dict, step=None):
        """Log metrics"""
        for key, value in metrics_dict.items():
            mlflow.log_metric(key, value, step=step)

    def log_model(self, model, model_name):
        """Log PyTorch model"""
        mlflow.pytorch.log_model(model, model_name)

def create_data_loaders():
    """Create CIFAR-10 data loaders"""
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    return train_loader, test_loader

def train_with_mlflow_tracking(model, model_name, epochs=3):
    """Train model with comprehensive MLflow tracking"""

    # Initialize tracker
    tracker = MLflowExperimentTracker("CNN_Architectures_Comparison")

    with mlflow.start_run(run_name=f"{model_name}_training"):
        # Log model metadata
        tracker.log_params({
            "model_name": model_name,
            "epochs": epochs,
            "batch_size": 32,
            "optimizer": "AdamW",
            "learning_rate": 1e-3,
            "num_parameters": sum(p.numel() for p in model.parameters())
        })

        # Setup
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        model = model.to(device)
        train_loader, test_loader = create_data_loaders()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

        best_accuracy = 0.0
        start_time = time.time()

        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            epoch_start = time.time()

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            # Calculate metrics
            train_accuracy = 100. * train_correct / train_total
            val_accuracy = 100. * val_correct / val_total
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(test_loader)
            epoch_time = time.time() - epoch_start

            # Log metrics
            tracker.log_metrics({
                "train_loss": avg_train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": avg_val_loss,
                "val_accuracy": val_accuracy,
                "epoch_time": epoch_time
            }, step=epoch)

            print(f"Epoch {epoch+1}/{epochs}: Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%")

            # Save best model
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                tracker.log_model(model, "best_model")

        total_time = time.time() - start_time

        # Log final metrics
        tracker.log_metrics({
            "best_accuracy": best_accuracy,
            "total_training_time": total_time,
            "final_train_accuracy": train_accuracy,
            "final_val_accuracy": val_accuracy
        })

        print(".2f")
        print(".2f")

        return best_accuracy

# 9.2 Model Versioning and Checkpointing
def save_model_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """Save model checkpoint with comprehensive metadata"""

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
        'timestamp': datetime.now().isoformat(),
        'model_architecture': model.__class__.__name__,
        'pytorch_version': torch.__version__,
        'num_parameters': sum(p.numel() for p in model.parameters())
    }

    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")

    # Save metadata separately
    metadata = {
        'filepath': filepath,
        'timestamp': checkpoint['timestamp'],
        'model_architecture': checkpoint['model_architecture'],
        'num_parameters': checkpoint['num_parameters'],
        'accuracy': accuracy,
        'epoch': epoch
    }

    metadata_file = filepath.replace('.pth', '_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

def load_model_checkpoint(filepath, model, optimizer=None):
    """Load model checkpoint with validation"""

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    checkpoint = torch.load(filepath, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', 0.0)
    accuracy = checkpoint.get('accuracy', 0.0)

    print(f"Checkpoint loaded from {filepath}")
    print(f"Epoch: {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")
    print(f"Model: {checkpoint.get('model_architecture', 'Unknown')}")
    print(f"Saved: {checkpoint.get('timestamp', 'Unknown')}")

    return epoch, loss, accuracy

# 9.3 Model Monitoring
class ModelMonitor:
    """Production model monitoring"""

    def __init__(self, model_name, accuracy_threshold=0.80):
        self.model_name = model_name
        self.accuracy_threshold = accuracy_threshold
        self.predictions = []
        self.max_history = 1000

    def log_prediction(self, prediction, actual=None, confidence=None):
        """Log prediction for monitoring"""

        entry = {
            'timestamp': datetime.now(),
            'prediction': prediction,
            'actual': actual,
            'confidence': confidence,
            'correct': prediction == actual if actual is not None else None
        }

        self.predictions.append(entry)

        # Maintain history size
        if len(self.predictions) > self.max_history:
            self.predictions = self.predictions[-self.max_history:]

    def get_accuracy(self, window_size=100):
        """Calculate recent accuracy"""

        if len(self.predictions) < window_size:
            return None

        recent = self.predictions[-window_size:]
        correct = sum(1 for p in recent if p['correct'] is True)

        return correct / len(recent)

    def check_performance_alert(self):
        """Check if model performance is below threshold"""

        accuracy = self.get_accuracy()

        if accuracy is None:
            return False, "Insufficient data"

        if accuracy < self.accuracy_threshold:
            return True, ".3f"

        return False, ".3f"

    def get_monitoring_report(self):
        """Generate monitoring report"""

        accuracy = self.get_accuracy()
        total_predictions = len(self.predictions)

        if total_predictions == 0:
            return {"status": "no_data"}

        # Calculate confidence statistics
        confidences = [p['confidence'] for p in self.predictions if p['confidence'] is not None]

        report = {
            "model_name": self.model_name,
            "total_predictions": total_predictions,
            "recent_accuracy": accuracy,
            "accuracy_threshold": self.accuracy_threshold,
            "avg_confidence": np.mean(confidences) if confidences else None,
            "min_confidence": np.min(confidences) if confidences else None,
            "max_confidence": np.max(confidences) if confidences else None,
            "generated_at": datetime.now().isoformat()
        }

        return report

# 9.4 FastAPI Model Server
class ModelServer:
    """Production-ready model server"""

    def __init__(self):
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.model = None
        self.monitor = ModelMonitor("CNN_Model")

        # CIFAR-10 class names
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                           'dog', 'frog', 'horse', 'ship', 'truck']

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def load_model(self, model_path, model_class):
        """Load trained model"""

        try:
            self.model = model_class(num_classes=10)
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()

            logger.info(f"Model loaded successfully from {model_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def predict(self, image: Image.Image) -> dict:
        """Make prediction on image"""

        try:
            # Preprocess
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # Inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)

            prediction = predicted_class.item()
            confidence_score = confidence.item()

            # Log for monitoring
            self.monitor.log_prediction(prediction, confidence=confidence_score)

            result = {
                'class_id': prediction,
                'class_name': self.class_names[prediction],
                'confidence': confidence_score,
                'probabilities': probabilities.squeeze().cpu().numpy().tolist(),
                'timestamp': datetime.now().isoformat()
            }

            return result

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

    def get_monitoring_report(self):
        """Get monitoring report"""
        return self.monitor.get_monitoring_report()

# Global model server instance
model_server = ModelServer()

# FastAPI app
app = FastAPI(title="CNN Model Serving API", version="1.0.0")

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    """Image classification endpoint"""

    try:
        # Validate file
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')

        # Make prediction
        result = model_server.predict(image)

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Prediction endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "model_loaded": model_server.model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/monitoring")
async def get_monitoring():
    """Get monitoring report"""
    return model_server.get_monitoring_report()

@app.get("/model_info")
async def model_info():
    """Get model information"""
    if model_server.model is None:
        return {"status": "no_model_loaded"}

    return {
        "model_loaded": True,
        "device": str(model_server.device),
        "num_parameters": sum(p.numel() for p in model_server.model.parameters()),
        "class_names": model_server.class_names
    }

# 9.5 Demo and Testing
def demo_mlops_pipeline():
    """Demonstrate MLOps pipeline"""

    print("MLOps Pipeline Demo")
    print("=" * 30)

    # Import model (assuming EfficientNet is available)
    from chapter8_modern_architectures_code import EfficientNet

    # Create model
    model = EfficientNet(num_classes=10)

    # Train with MLflow tracking
    print("\\n1. Training with MLflow tracking...")
    try:
        accuracy = train_with_mlflow_tracking(model, "EfficientNet_Demo", epochs=1)
        print(".2f")
    except Exception as e:
        print(f"MLflow training failed: {e}")

    # Save checkpoint
    print("\\n2. Saving model checkpoint...")
    try:
        save_model_checkpoint(
            model, None, 1, 0.5, accuracy,
            "models/efficientnet_checkpoint.pth"
        )
    except Exception as e:
        print(f"Checkpoint saving failed: {e}")

    # Load checkpoint
    print("\\n3. Loading model checkpoint...")
    try:
        new_model = EfficientNet(num_classes=10)
        _, _, _ = load_model_checkpoint(
            "models/efficientnet_checkpoint.pth", new_model
        )
    except Exception as e:
        print(f"Checkpoint loading failed: {e}")

    # Model monitoring demo
    print("\\n4. Model monitoring demo...")
    monitor = ModelMonitor("Demo_Model")

    # Simulate predictions
    rng = np.random.default_rng(42)  # For reproducible results
    for _ in range(10):
        pred = rng.integers(0, 10)
        actual = rng.integers(0, 10)
        conf = rng.random()
        monitor.log_prediction(pred, actual, conf)

    accuracy = monitor.get_accuracy(window_size=10)
    print(".3f")

    alert, message = monitor.check_performance_alert()
    print(f"Alert: {alert}, Message: {message}")

    print("\\n5. Model server setup...")
    # Note: Server would be started separately
    print("FastAPI server would be available at http://localhost:8000")
    print("Use: uvicorn chapter9_mlops_deployment_code:app --reload")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "server":
        # Run FastAPI server
        print("Starting FastAPI server...")
        print("Load your model first by calling model_server.load_model()")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        # Run demo
        demo_mlops_pipeline()