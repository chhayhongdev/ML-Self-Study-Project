#!/usr/bin/env python3
"""
CNN Model Inference Script
Load the trained ResNet18 model and make predictions on images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Using device: {device}')

# CIFAR-10 classes
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# ResNet Basic Block
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# ResNet Model
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

# Load the trained model
def load_model(model_path='cnn_cifar10.pth'):
    model = ResNet18()
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"✅ Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

# Image preprocessing
def preprocess_image(image):
    """Preprocess image for CIFAR-10 model (32x32)"""
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    return transform(image).unsqueeze(0).to(device)

# Prediction function
def predict_image(model, image_tensor):
    """Make prediction on preprocessed image tensor"""
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_idx = probabilities.max(1)

    predicted_class = classes[predicted_idx.item()]
    confidence_score = confidence.item()

    return predicted_class, confidence_score, probabilities.squeeze().cpu().numpy()

# Load image from file
def load_image_from_file(file_path):
    """Load image from local file"""
    try:
        image = Image.open(file_path).convert('RGB')
        return image
    except Exception as e:
        print(f"❌ Error loading image from {file_path}: {e}")
        return None

# Load image from URL
def load_image_from_url(url):
    """Load image from URL"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert('RGB')
        return image
    except Exception as e:
        print(f"❌ Error loading image from URL: {e}")
        return None

# Display prediction
def display_prediction(image, predicted_class, confidence):
    """Display image with prediction"""
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.title(f'Predicted: {predicted_class}\nConfidence: {confidence:.2f}')
    plt.axis('off')
    plt.show()

# Main inference functions
def predict_from_file(model, file_path, show_image=True):
    """Predict from local file"""
    image = load_image_from_file(file_path)
    if image is None:
        return

    processed_image = preprocess_image(image)
    predicted_class, confidence, probs = predict_image(model, processed_image)

    print(f"📁 File: {file_path}")
    print(f"🎯 Prediction: {predicted_class}")
    print(f"📊 Confidence: {confidence:.2f}")

    if show_image:
        display_prediction(image, predicted_class, confidence)

    return predicted_class, confidence, probs

def predict_from_url(model, url, show_image=True):
    """Predict from URL"""
    image = load_image_from_url(url)
    if image is None:
        return

    processed_image = preprocess_image(image)
    predicted_class, confidence, probs = predict_image(model, processed_image)

    print(f"🌐 URL: {url}")
    print(f"🎯 Prediction: {predicted_class}")
    print(f"📊 Confidence: {confidence:.2f}")

    if show_image:
        display_prediction(image, predicted_class, confidence)

    return predicted_class, confidence, probs

# Example usage
if __name__ == "__main__":
    # Load model
    model = load_model()
    if model is None:
        exit(1)

    print("\n🚀 Model ready for inference!")
    print("📋 Available classes:", classes)

    # Example predictions
    print("\n" + "="*50)
    print("🔍 EXAMPLE PREDICTIONS")
    print("="*50)

    # Example 1: URL prediction
    example_urls = [
        "https://www.thesprucepets.com/thmb/yH_dVf9ehigKYxqsvP5IHKviUKQ=/724x0/filters:no_upscale():strip_icc()/GettyImages-1313232209-e412c4dc9411489f8197c9c0067c94ed.jpg",
        "https://images.unsplash.com/photo-1518709268805-4e9042af2176?w=400",
        "https://images.unsplash.com/photo-1444927714506-8492d94c9bd1?w=400"
    ]

    for i, url in enumerate(example_urls, 1):
        print(f"\n📸 Example {i}:")
        predict_from_url(model, url, show_image=False)

    print("\n" + "="*50)
    print("💡 USAGE EXAMPLES")
    print("="*50)
    print("""
# Predict from local file:
# predict_from_file(model, 'path/to/your/image.jpg')

# Predict from URL:
# predict_from_url(model, 'https://example.com/image.jpg')

# Get raw probabilities:
# predicted_class, confidence, probabilities = predict_from_url(model, url)
# print("All probabilities:", probabilities)
    """)