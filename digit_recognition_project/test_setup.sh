#!/bin/bash
# Test script to verify the CNN Digit Recognition setup

echo "🧪 Testing CNN Digit Recognition Setup"
echo "======================================"
echo ""

# Check if Docker containers are running
echo "🐳 Checking Docker containers..."
if docker-compose ps | grep -q "Up"; then
    echo "✅ Docker containers are running"
else
    echo "❌ Docker containers are not running"
    echo "   Run: docker-compose up -d"
    exit 1
fi

echo ""

# Test health endpoint
echo "🏥 Testing API health..."
if curl -s http://localhost:8000/health | grep -q "healthy"; then
    echo "✅ API health check passed"
else
    echo "❌ API health check failed"
    exit 1
fi

echo ""

# Test web interface accessibility
echo "🌐 Testing web interface..."
if curl -s http://localhost:8000/web_interface/index.html | grep -q "<!DOCTYPE html>"; then
    echo "✅ Web interface is accessible"
else
    echo "❌ Web interface is not accessible"
    exit 1
fi

echo ""

# Test API prediction endpoint
echo "🤖 Testing prediction API..."
RESPONSE=$(curl -s -X POST http://localhost:8000/predict-base64 \
  -H "Content-Type: application/json" \
  -d '{"image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="}')

if echo "$RESPONSE" | grep -q '"success":true'; then
    echo "✅ Prediction API is working"
    PREDICTION=$(echo "$RESPONSE" | grep -o '"prediction":[0-9]' | cut -d':' -f2)
    CONFIDENCE=$(echo "$RESPONSE" | grep -o '"confidence":[0-9.]*' | cut -d':' -f2 | head -1)
    echo "   Sample prediction: Digit $PREDICTION (confidence: $CONFIDENCE)"
else
    echo "❌ Prediction API failed"
    echo "   Response: $RESPONSE"
    exit 1
fi

echo ""
echo "🎉 All tests passed! Your CNN Digit Recognition system is working correctly."
echo ""
echo "🌐 Access your application:"
echo "   Web Interface: http://localhost:8000/web_interface/index.html"
echo "   API Docs:      http://localhost:8000/docs"
echo ""
echo "📊 View logs: docker-compose logs -f"
echo "🛑 Stop app:  docker-compose down"