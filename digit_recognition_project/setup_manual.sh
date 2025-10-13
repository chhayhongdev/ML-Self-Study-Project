#!/bin/bash
# Manual setup script for CNN Digit Recognition Project
# For users who prefer not to use Docker

set -e

echo "🛠️  Manual Setup for CNN Digit Recognition Project"
echo "================================================="
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "🐍 Python version: $PYTHON_VERSION"

# Check if Python 3.8+ is available
PYTHON_MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")
PYTHON_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")

if [ "$PYTHON_MAJOR" -lt 3 ] || [ "$PYTHON_MINOR" -lt 8 ]; then
    echo "❌ Python 3.8+ is required. Current version: $PYTHON_VERSION"
    echo "   Please upgrade Python or use the Docker setup instead."
    exit 1
fi

echo "✅ Python version is compatible"
echo ""

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
echo "✅ Virtual environment created"
echo ""

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate
echo "✅ Virtual environment activated"
echo ""

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip
echo "✅ Pip upgraded"
echo ""

# Install dependencies
echo "📥 Installing dependencies..."
echo "This may take a few minutes..."
pip install -r requirements.txt
echo "✅ Dependencies installed"
echo ""

# Check if model file exists
if [ ! -f "digit_classifier.pth" ]; then
    echo "🤖 Training model (this may take 5-10 minutes)..."
    python3 cnn_digit_recognition_project.py
    echo "✅ Model trained"
else
    echo "✅ Pre-trained model found"
fi

echo ""
echo "🚀 Starting web application..."
echo "   Press Ctrl+C to stop"
echo ""
echo "🌐 Once started, access:"
echo "   Web Interface: http://localhost:8000/web_interface/index.html"
echo "   API Health:    http://localhost:8000/health"
echo ""

# Start the application
cd web_interface
python3 app.py