# Cloud Deployment Extension
# Future extension: Deploy model to cloud platforms

"""
This folder will contain cloud deployment scripts and configurations.

Cloud Deployment Options:
1. AWS Lambda + API Gateway
2. Google Cloud Functions
3. Azure Functions
4. Docker containers for any cloud
5. Serverless deployment

Benefits:
- Scalable inference
- Pay-per-use pricing
- Global distribution
- Easy integration with web/mobile apps

Future implementations:
1. Docker containerization
2. REST API with FastAPI/Flask
3. CloudFormation/SAM templates
4. CI/CD pipelines
5. Monitoring and logging
"""

import json

def create_dockerfile():
    """
    Future: Generate Dockerfile for containerized deployment
    """
    dockerfile_content = """
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and application
COPY models/ ./models/
COPY app.py .

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
"""
    return dockerfile_content

# Cloud Deployment Implementation
# Complete Docker and cloud deployment solution

"""
Production-ready deployment scripts for cloud platforms.

This module provides:
1. Docker containerization
2. FastAPI web service
3. Docker Compose orchestration
4. Nginx reverse proxy
5. Health checks and monitoring
6. Cloud platform deployment scripts
"""

import json
import os
from pathlib import Path

def create_dockerfile():
    """Generate production Dockerfile"""
    dockerfile_content = """# Multi-stage Docker build for Digit Recognition API

# Stage 1: Build stage with PyTorch and dependencies
FROM python:3.11-slim AS builder

# Install system dependencies for PyTorch
RUN apt-get update && apt-get install -y \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime stage (smaller final image)
FROM python:3.11-slim

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \\
    libgomp1 \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \\
    && mkdir -p /app \\
    && chown -R app:app /app

# Set working directory
WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code and model
COPY web_interface/app.py .
COPY digit_classifier.pth* ./

# Change to non-root user
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "app.py"]
"""
    return dockerfile_content

def create_docker_compose():
    """Generate docker-compose.yml for local deployment"""
    compose_content = """version: '3.8'

services:
  digit-recognition-api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - PYTHONUNBUFFERED=1
    volumes:
      # Mount model files (optional, can be baked into image)
      - ./digit_classifier.pth:/app/digit_classifier.pth:ro
      - ./digit_classifier_metadata.json:/app/digit_classifier_metadata.json:ro
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  # Optional: Nginx reverse proxy for production
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - digit-recognition-api
    restart: unless-stopped
    profiles:
      - production
"""
    return compose_content

def create_deployment_script():
    """Generate deployment script for different cloud platforms"""
    script_content = """#!/bin/bash
# Cloud Deployment Script for Digit Recognition API

set -e

echo "🚀 Digit Recognition API - Cloud Deployment"
echo "=========================================="

# Colors for output
RED='\\033[0;31m'
GREEN='\\033[0;32m'
YELLOW='\\033[1;33m'
NC='\\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    print_status "Docker is installed"
}

# Check if model files exist
check_model_files() {
    if [ ! -f "digit_classifier.pth" ]; then
        print_error "Model file digit_classifier.pth not found. Please train the model first."
        exit 1
    fi
    print_status "Model files found"
}

# Build Docker image
build_image() {
    echo "Building Docker image..."
    docker build -t digit-recognition-api .
    print_status "Docker image built successfully"
}

# Run locally with Docker Compose
deploy_local() {
    echo "Starting local deployment with Docker Compose..."
    docker-compose up -d
    print_status "Application started locally"

    echo "Waiting for health check..."
    sleep 10

    # Check if service is healthy
    if curl -f http://localhost:8000/health &>/dev/null; then
        print_status "Service is healthy!"
        echo "🌐 API available at: http://localhost:8000"
        echo "🖼️  Test page at: http://localhost:8000/static/index.html"
    else
        print_error "Service health check failed"
        exit 1
    fi
}

# Deploy to production with Nginx
deploy_production() {
    echo "Starting production deployment with Nginx..."
    docker-compose --profile production up -d
    print_status "Production deployment started"

    echo "Waiting for services..."
    sleep 15

    # Check services
    if curl -f http://localhost/health &>/dev/null; then
        print_status "Production deployment successful!"
        echo "🌐 API available at: http://localhost"
    else
        print_error "Production deployment health check failed"
        exit 1
    fi
}

# Show usage
usage() {
    echo "Usage: $0 [local|production|build|stop]"
    echo ""
    echo "Commands:"
    echo "  local      - Deploy locally with Docker Compose"
    echo "  production - Deploy with Nginx reverse proxy"
    echo "  build      - Build Docker image only"
    echo "  stop       - Stop all running containers"
    echo ""
    echo "Examples:"
    echo "  ./deploy.sh local      # Start local development"
    echo "  ./deploy.sh production # Start production with Nginx"
    echo "  ./deploy.sh build      # Build image only"
    echo "  ./deploy.sh stop       # Stop everything"
}

# Stop all containers
stop_containers() {
    echo "Stopping all containers..."
    docker-compose down 2>/dev/null || true
    docker-compose --profile production down 2>/dev/null || true
    print_status "All containers stopped"
}

# Main deployment logic
main() {
    case "${1:-local}" in
        "local")
            check_docker
            check_model_files
            build_image
            deploy_local
            ;;
        "production")
            check_docker
            check_model_files
            build_image
            deploy_production
            ;;
        "build")
            check_docker
            check_model_files
            build_image
            ;;
        "stop")
            stop_containers
            ;;
        *)
            usage
            exit 1
            ;;
    esac
}

# Run main function
main "$@"

if __name__ == "__main__":
    print("Cloud deployment extension not yet implemented.")
    print("Future: Will support deployment to AWS, GCP, Azure")
    print("Features: Docker containers, REST APIs, serverless functions")