# FastAPI Web Service for Digit Recognition
# Production-ready API for model deployment

"""
This module creates a FastAPI web service for the digit recognition model.
Features:
- REST API endpoints for prediction
- Health check endpoint
- Model metadata endpoint
- CORS support for web applications
- Proper error handling
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
import base64
import json
import time
from typing import Dict, Any
import uvicorn
import aiofiles

# Initialize FastAPI app
app = FastAPI(
    title="Digit Recognition API",
    description="CNN-based handwritten digit recognition service",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=".", html=True), name="static")

# Device configuration
device = torch.device('mps' if torch.backends.mps.is_available() else
                     'cuda' if torch.cuda.is_available() else 'cpu')

# Global model variable
model = None

class DigitClassifier(nn.Module):
    """CNN model for digit classification"""

    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def load_model():
    """Load the trained model"""
    global model
    try:
        model = DigitClassifier()
        model.load_state_dict(torch.load('digit_classifier.pth', map_location=device, weights_only=False)['model_state_dict'])
        model.to(device)
        model.eval()
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        raise

def preprocess_image(image_data: bytes) -> torch.Tensor:
    """Preprocess image for model prediction"""
    try:
        # Open image
        image = Image.open(io.BytesIO(image_data)).convert('L')  # Convert to grayscale

        # Invert colors (canvas is black-on-white, MNIST is white-on-black)
        image = Image.eval(image, lambda x: 255 - x)

        # Resize to 28x28
        image = image.resize((28, 28))

        # Convert to tensor and normalize
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        tensor = transform(image)
        return tensor

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

def predict_digit(image_tensor: torch.Tensor) -> Dict[str, Any]:
    """Make prediction on preprocessed image"""
    global model

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Add batch dimension
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)

        # Get top 3 predictions
        top3_prob, top3_class = torch.topk(probabilities, 3, dim=1)

        result = {
            "prediction": int(predicted_class.item()),
            "confidence": float(confidence.item()),
            "top3_predictions": [
                {"digit": int(cls.item()), "confidence": float(prob.item())}
                for cls, prob in zip(top3_class[0], top3_prob[0])
            ]
        }

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Digit Recognition API",
        "version": "1.0.0",
        "endpoints": {
            "GET /": "This information",
            "GET /health": "Health check",
            "GET /model-info": "Model metadata",
            "POST /predict": "Predict digit from image",
            "POST /predict-base64": "Predict digit from base64 image"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "model_loaded": model is not None,
        "device": str(device)
    }

@app.get("/model-info")
async def model_info():
    """Get model information"""
    try:
        async with aiofiles.open('digit_classifier_metadata.json', 'r') as f:
            content = await f.read()
            metadata = json.loads(content)
        return metadata
    except FileNotFoundError:
        return {"error": "Model metadata not found"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading metadata: {str(e)}")

@app.post("/predict")
async def predict_from_file(file: UploadFile = File(...)):
    """Predict digit from uploaded image file"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_data = await file.read()
        image_tensor = preprocess_image(image_data)
        result = predict_digit(image_tensor)

        return JSONResponse(content={
            "success": True,
            "result": result,
            "filename": file.filename
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-base64")
async def predict_from_base64(data: Dict[str, str]):
    """Predict digit from base64 encoded image"""
    try:
        if "image" not in data:
            raise HTTPException(status_code=400, detail="Missing 'image' field")

        # Decode base64
        image_data = base64.b64decode(data["image"])
        image_tensor = preprocess_image(image_data)
        result = predict_digit(image_tensor)

        return JSONResponse(content={
            "success": True,
            "result": result
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/web_interface/index.html")
async def web_interface():
    """Serve the web interface"""
    try:
        async with aiofiles.open("index.html", "r") as f:
            html_content = await f.read()
        from fastapi.responses import HTMLResponse
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Web interface not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving web interface: {str(e)}")

@app.get("/web_interface/")
async def web_interface_root():
    """Redirect to web interface"""
    return await web_interface()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)