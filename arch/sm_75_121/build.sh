#!/usr/bin/env bash
set -euo pipefail

VERSION="$(cat "$(dirname "${BASH_SOURCE[0]}")/../../VERSION")"
# Version and arch both matter: two arch builds exist and .env picks one.
IMAGE_TAG="frigate-inference:v${VERSION}-sm_75_121"
# Moving tag, so existing .env entries keep working.
ARCH_TAG="frigate-inference:sm_75_121"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "Building ${IMAGE_TAG}..."
cd "${BUILD_ROOT}"
docker build \
    --file arch/sm_75_121/Dockerfile \
    --tag "${IMAGE_TAG}" \
    --tag "${ARCH_TAG}" \
    .
echo "Done: ${IMAGE_TAG} (also tagged ${ARCH_TAG})"
