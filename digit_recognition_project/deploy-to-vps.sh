#!/usr/bin/env bash
set -euo pipefail

# Deploy Digit Recognition API to VPS Script
# Builds the API image locally, saves to tar.gz, copies to VPS, and deploys
# Usage: ./deploy-to-vps.sh <vps_user> <vps_ip>
# Example: ./deploy-to-vps.sh chhayhong 157.10.73.155

VPS_USER=${1:-}
VPS_IP=${2:-}

# Helper: ask for confirmation. Returns 0 for yes, 1 for no.
ask_confirm() {
  local prompt="$1"
  local reply
  while true; do
    read -r -p "$prompt [y/N]: " reply
    case "$reply" in
      [Yy]) return 0 ;;
      [Nn] | "") return 1 ;;
      *) echo "Please answer y or n." ;;
    esac
  done
}

if [ -z "$VPS_USER" ] || [ -z "$VPS_IP" ]; then
  cat <<USAGE
Usage: $0 <vps_user> <vps_ip>

Example:
  $0 chhayhong 157.10.73.155

Notes:
  - Builds the digit-recognition-api image locally
  - Saves image to compressed tar.gz file
  - Uploads to VPS and deploys the service
  - Uses docker-compose for deployment on VPS
USAGE
  exit 1
fi

echo "🚀 Deploying Digit Recognition API to $VPS_USER@$VPS_IP"

# Image names
API_IMAGE="digit_recognition_project-digit-recognition-api:latest"

# Tar files
API_TAR="digit-recognition-api-image.tar.gz"

DEPLOY_DIR="/home/${VPS_USER}/digit-recognition-deploy"

echo "🏗️ Building digit-recognition-api Docker image locally (linux/amd64)..."
if ask_confirm "Build API image locally?"; then
  echo "Building API image..."
  docker build --platform linux/amd64 -t ${API_IMAGE} .
else
  echo "Skipping build step."
  exit 0
fi

echo "💾 Saving API image to compressed tar file..."
docker save ${API_IMAGE} | gzip > "${API_TAR}"

echo "📤 Copying API image to VPS: ${VPS_USER}@${VPS_IP}:${DEPLOY_DIR}"
ssh ${VPS_USER}@${VPS_IP} "mkdir -p ${DEPLOY_DIR}"
scp ${API_TAR} ${VPS_USER}@${VPS_IP}:${DEPLOY_DIR}/

# Copy deployment files
echo "📋 Copying deployment files to VPS..."
scp docker-compose.yml nginx.conf ${VPS_USER}@${VPS_IP}:${DEPLOY_DIR}/

echo "🐳 Loading API image and deploying on VPS..."
ssh ${VPS_USER}@${VPS_IP} "DEPLOY_DIR='${DEPLOY_DIR}' API_TAR='${API_TAR}' API_IMAGE='${API_IMAGE}' bash -s" <<'EOF'
set -xeo pipefail
cd "$DEPLOY_DIR"

echo '📦 Checking uploaded files on VPS...'
ls -la ${API_TAR} docker-compose.yml nginx.conf || true

# Determine whether docker is callable without passwordless sudo
SUDO=''
if docker info >/dev/null 2>&1; then
  SUDO=''
elif sudo -n true 2>/dev/null; then
  SUDO='sudo'
else
  echo "ERROR: Docker requires sudo with password on the remote host."
  echo "Either make the remote user part of the 'docker' group or allow passwordless sudo for docker commands."
  exit 1
fi

echo '📦 Loading digit-recognition-api Docker image on VPS...'
if [ -f ${API_TAR} ]; then
  echo "Loading ${API_TAR}..."
  gunzip -c ${API_TAR} | ${SUDO} docker load
else
  echo "Error: ${API_TAR} not found"
  exit 1
fi

echo '🧹 Cleaning up compressed tar file on VPS...'
rm -f ${API_TAR}

# Create unique timestamped tag to force service update
TS=$(date -u +%Y%m%d%H%M%S)
echo "🔖 Tagging API image with timestamp: ${TS}"
if ${SUDO} docker image inspect ${API_IMAGE} >/dev/null 2>&1; then
  API_NEW="${API_IMAGE%:*}:${TS}"
  ${SUDO} docker tag ${API_IMAGE} ${API_NEW}
  echo "Tagged ${API_IMAGE} -> ${API_NEW}"
else
  echo "Error: ${API_IMAGE} not found after loading"
  exit 1
fi

echo '🐳 Starting/updating services with docker-compose...'
# Update the image in docker-compose.yml temporarily
sed -i.bak "s|image:.*|image: ${API_NEW}|" docker-compose.yml

# Deploy with docker-compose
${SUDO} docker-compose down || true
${SUDO} docker-compose up -d

echo '⏳ Waiting for services to be healthy (timeout 120s)...'
timeout=120
interval=5
elapsed=0
healthy=0

while [ $elapsed -lt $timeout ]; do
  # Check if digit-recognition-api is healthy
  if curl -f http://localhost:8000/health >/dev/null 2>&1; then
    healthy=1
    break
  fi

  sleep $interval
  elapsed=$((elapsed + interval))
done

if [ $healthy -eq 1 ]; then
  echo '✅ Services deployed successfully!'
  ${SUDO} docker-compose ps
else
  echo '⚠️ Timeout waiting for services to become healthy.'
  ${SUDO} docker-compose ps
  echo 'Check service logs:'
  ${SUDO} docker-compose logs --tail=20
fi

echo '📊 Current service status:'
${SUDO} docker-compose ps

EOF

echo "🧹 Cleaning up local tar file..."
rm -f ${API_TAR}

echo "🎉 Digit Recognition API deployment finished!"
echo "📝 Check API logs: ssh ${VPS_USER}@${VPS_IP} 'cd ${DEPLOY_DIR} && sudo docker-compose logs -f'"
echo "🔍 Check API status: ssh ${VPS_USER}@${VPS_IP} 'cd ${DEPLOY_DIR} && sudo docker-compose ps'"
echo "🌐 API should be available at: http://${VPS_IP}:8000"
echo "🌐 With nginx (if enabled): http://${VPS_IP}"

exit 0