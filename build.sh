#!/bin/bash

# Modern Docker build script for TokenGuard
set -euo pipefail

# Build arguments
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
VCS_REF=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
VERSION=${1:-latest}

echo "Building TokenGuard Docker image..."
echo "Build Date: $BUILD_DATE"
echo "VCS Ref: $VCS_REF"
echo "Version: $VERSION"

# Build with modern features
docker build \
  --platform linux/amd64,linux/arm64 \
  --build-arg BUILD_DATE="$BUILD_DATE" \
  --build-arg VCS_REF="$VCS_REF" \
  --build-arg VERSION="$VERSION" \
  --progress=plain \
  --tag "tokenguard:$VERSION" \
  --tag "tokenguard:latest" \
  .

echo "Build completed successfully!"
echo "Run with: docker run -p 8000:8000 --env-file .env tokenguard:latest"
