#!/bin/bash
# CNN Digit Recognition Project - Setup Script
# This script helps you set up the project from scratch

set -e  # Exit on any error

echo "🎯 CNN Digit Recognition Project Setup"
echo "======================================"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first:"
    echo "   https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first:"
    echo "   https://docs.docker.com/compose/install/"
    exit 1
fi

echo "✅ Docker and Docker Compose are installed"

# Check available disk space (rough estimate)
DISK_SPACE=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$DISK_SPACE" -lt 2 ]; then
    echo "⚠️  Warning: Only ${DISK_SPACE}GB free disk space detected."
    echo "   This project needs ~2GB for Docker images and datasets."
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "🚀 Starting setup process..."
echo ""

# Stop any existing containers
echo "🛑 Stopping any existing containers..."
docker-compose down 2>/dev/null || true

# Remove old images (optional)
read -p "🧹 Remove old Docker images to save space? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Removing unused Docker images..."
    docker image prune -f
fi

echo ""
echo "🏗️  Building Docker containers..."
echo "This may take 5-10 minutes depending on your internet connection..."
echo ""

# Build and start containers
if docker-compose up --build -d; then
    echo ""
    echo "✅ Setup completed successfully!"
    echo ""
    echo "🌐 Access your application:"
    echo "   Web Interface: http://localhost:8000/web_interface/index.html"
    echo "   API Health:    http://localhost:8000/health"
    echo "   API Docs:      http://localhost:8000/docs"
    echo ""
    echo "📊 To view logs: docker-compose logs -f"
    echo "🛑 To stop:      docker-compose down"
    echo ""
    echo "🎨 Try drawing some digits on the web interface!"
    echo ""
else
    echo ""
    echo "❌ Setup failed. Common issues:"
    echo "   - Not enough RAM (need 4GB+)"
    echo "   - Slow internet connection"
    echo "   - Docker daemon not running"
    echo ""
    echo "🔧 Troubleshooting:"
    echo "   - Check Docker logs: docker-compose logs"
    echo "   - Restart Docker daemon"
    echo "   - Free up disk space"
    echo ""
    exit 1
fi