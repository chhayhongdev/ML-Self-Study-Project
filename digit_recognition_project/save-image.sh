#!/usr/bin/env bash
set -euo pipefail

# Save Digit Recognition API Docker Image to tar.gz
# Usage: ./save-image.sh [image_name] [output_file]
# Example: ./save-image.sh digit_recognition_project-digit-recognition-api:latest digit-recognition-api.tar.gz

IMAGE_NAME=${1:-"digit_recognition_project-digit-recognition-api:latest"}
OUTPUT_FILE=${2:-"digit-recognition-api-image.tar.gz"}

echo "💾 Saving Docker image '$IMAGE_NAME' to '$OUTPUT_FILE'..."

# Check if image exists
if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
  echo "❌ Error: Image '$IMAGE_NAME' not found locally."
  echo "   Build the image first with: docker-compose build"
  exit 1
fi

# Save and compress the image
echo "📦 Compressing image (this may take a few minutes)..."
docker save "$IMAGE_NAME" | gzip > "$OUTPUT_FILE"

# Get file size
FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)

echo "✅ Image saved successfully!"
echo "   File: $OUTPUT_FILE"
echo "   Size: $FILE_SIZE"
echo ""
echo "🚀 To deploy to VPS, run:"
echo "   ./deploy-to-vps.sh <vps_user> <vps_ip>"