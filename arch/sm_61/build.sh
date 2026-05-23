#!/usr/bin/env bash
set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────
TORCH_TAG="v2.5.1"
VISION_TAG="v0.20.1"
CUDA_ARCH="6.1"
BUILD_IMAGE="nvidia/cuda:12.2.2-devel-ubuntu22.04"
IMAGE_TAG="frigate-inference:sm_61"

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE="${BUILD_ROOT}/pytorch-workspace"
SRC="${WORKSPACE}/src"

# ── PyTorch wheel build ────────────────────────────────────────────────────
if ls "${SRC}"/torch-*.whl &>/dev/null && ls "${SRC}"/torchvision-*.whl &>/dev/null; then
    echo "Wheels found in ${SRC} — skipping PyTorch build."
else
    echo "Building PyTorch ${TORCH_TAG} and Torchvision ${VISION_TAG} for sm_${CUDA_ARCH}..."

    mkdir -p "${SRC}" \
             "${WORKSPACE}/pip-cache" \
             "${WORKSPACE}/ccache" \
             "${WORKSPACE}/build-torch" \
             "${WORKSPACE}/build-vision"

    [[ -d "${SRC}/pytorch" ]] || \
        git clone --recursive --branch "${TORCH_TAG}" \
            https://github.com/pytorch/pytorch.git "${SRC}/pytorch"

    [[ -d "${SRC}/vision" ]] || \
        git clone --recursive --branch "${VISION_TAG}" \
            https://github.com/pytorch/vision.git "${SRC}/vision"

    docker run --rm --gpus all -i \
        -v "${SRC}:/workspace" \
        -v "${WORKSPACE}/pip-cache:/root/.cache/pip" \
        -v "${WORKSPACE}/ccache:/root/.cache/ccache" \
        -v "${WORKSPACE}/build-torch:/workspace/pytorch/build" \
        -v "${WORKSPACE}/build-vision:/workspace/vision/build" \
        -e "CUDA_ARCH=${CUDA_ARCH}" \
        -e "MAX_JOBS=$(nproc)" \
        "${BUILD_IMAGE}" bash << 'EOF'
set -eux

apt-get update -q && apt-get install -y -q --no-install-recommends \
    git build-essential python3-dev python3-pip cmake ccache \
    libjpeg-dev libpng-dev

export TORCH_CUDA_ARCH_LIST="${CUDA_ARCH}"
export FORCE_CUDA=1
export TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
export CCACHE_DIR=/root/.cache/ccache
export CCACHE_MAXSIZE=20G
export WITH_CCACHE=1

echo "--- Building PyTorch (MAX_JOBS=${MAX_JOBS}) ---"
cd /workspace/pytorch
[[ -f setup.py ]] || git submodule update --init --recursive
pip3 install -r requirements.txt
python3 setup.py bdist_wheel
pip3 install dist/*.whl
cp dist/*.whl /workspace/

echo "--- Building Torchvision ---"
cd /workspace/vision
pip3 install numpy pillow
export FORCE_NVCC=1
python3 setup.py bdist_wheel
cp dist/*.whl /workspace/

echo "--- Wheels written to /workspace ---"
ls -lh /workspace/*.whl
EOF

    echo "Wheel build complete."
fi

# ── Container build ────────────────────────────────────────────────────────
echo "Building ${IMAGE_TAG}..."
cd "${BUILD_ROOT}"
docker build \
    --file arch/sm_61/Dockerfile \
    --tag "${IMAGE_TAG}" \
    .
echo "Done: ${IMAGE_TAG}"
