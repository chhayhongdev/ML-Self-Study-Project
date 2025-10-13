#!/bin/bash
# Cloud Deployment Script for Digit Recognition API

set -e

echo "🚀 Digit Recognition API - Cloud Deployment"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

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
        echo "🖼️  Test page available in web_interface/index.html"
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