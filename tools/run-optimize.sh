#!/usr/bin/env bash
# Compile a TensorRT engine and run the inference benchmark.
#
# Spins up a fresh inference server, waits for it to be ready, then runs
# optimize.py inside a second container against the same volumes. Both
# containers are torn down on exit and server logs are saved.
#
# Usage:
#   ./tools/run-optimize.sh <model_name> [--test-only]
#
# Examples:
#   ./tools/run-optimize.sh yolo26n
#   ./tools/run-optimize.sh yolo26n --test-only
#   ./tools/run-optimize.sh your-frigate-plus-model
set -euo pipefail

IMAGE="${INFERENCE_IMAGE:-frigate-inference:sm_61}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ZMQ_DIR="$SCRIPT_DIR/zmq"
MODELS_DIR="$ROOT_DIR/models"
CONFIG_DIR="$ROOT_DIR/config"

if [[ -z "${1:-}" || "$1" == "--test-only" && -z "${2:-}" ]]; then
    echo "Usage: $(basename "$0") <model_name> [--test-only]"
    echo ""
    echo "Models in $MODELS_DIR:"
    ls -1 "$MODELS_DIR" 2>/dev/null | grep -v '\.engine$' || echo "  (none)"
    exit 1
fi

MODEL_NAME="$1"
TEST_FLAG=""
if [[ "${2:-}" == "--test-only" ]]; then
    TEST_FLAG="--test-only"
fi

mkdir -p "$ZMQ_DIR" "$MODELS_DIR"
chmod 777 "$ZMQ_DIR" "$MODELS_DIR"

# ── Start inference server ─────────────────────────────────────────────────
echo "==> Starting inference server ($IMAGE)..."
CONTAINER_ID=$(docker run -d \
    --name frigate-inference-optimize \
    --gpus all \
    -v "$ZMQ_DIR:/run/zmq" \
    -v "$MODELS_DIR:/models" \
    -v "$CONFIG_DIR:/config:ro" \
    "$IMAGE")

cleanup() {
    echo ""
    echo "==> Server logs (last 20 lines):"
    docker logs "$CONTAINER_ID" --tail 20
    docker logs "$CONTAINER_ID" > "$SCRIPT_DIR/server_last_run.log" 2>&1
    echo "(Full logs saved to tools/server_last_run.log)"
    docker rm -f "$CONTAINER_ID" >/dev/null 2>&1
    rm -f "$ZMQ_DIR/detector.sock"
}
trap cleanup EXIT

# ── Wait for ZMQ socket ────────────────────────────────────────────────────
echo "==> Waiting for ZMQ socket (up to 120s)..."
for i in $(seq 1 120); do
    [[ -S "$ZMQ_DIR/detector.sock" ]] && { echo "    Ready after ${i}s."; break; }
    [[ "$i" -eq 120 ]] && { echo "[ERROR] Socket never appeared."; exit 1; }
    sleep 1
done

# ── Run optimizer ──────────────────────────────────────────────────────────
echo "==> Running optimize.py..."
docker run --rm \
    --gpus all \
    -v "$SCRIPT_DIR/optimize.py:/optimize.py:ro" \
    -v "$ZMQ_DIR:/run/zmq" \
    -v "$MODELS_DIR:/models" \
    -v "$CONFIG_DIR:/config:ro" \
    -v "$(pwd)/image.png:/test_client_image.png:ro" \
    "$IMAGE" \
    python3 /optimize.py "$MODEL_NAME" $TEST_FLAG

echo "==> Done."
