#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="frigate-inference:sm_75plus"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "Building ${IMAGE_TAG}..."
cd "${BUILD_ROOT}"
docker build \
    --file arch/sm_75plus/Dockerfile \
    --tag "${IMAGE_TAG}" \
    .
echo "Done: ${IMAGE_TAG}"
