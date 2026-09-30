FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN --mount=type=cache,id=apt-cache-u24,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=apt-lists-u24,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 python3-pip \
        libgl1 libglib2.0-0 libgomp1

WORKDIR /app

# Update --index-url when PyTorch publishes wheels for a newer CUDA version.
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    pip3 install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128 \
    --break-system-packages

RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    pip3 install \
# tensorrt is pinned: an unpinned rebuild silently jumps TRT major version,
# which changes the builder API *and* invalidates every cached .engine
# (engines are not portable across TRT serialization versions).
# Changing this pin requires reviewing PASCAL_COMPAT codepaths in yolo_engine.py.
    tensorrt==11.3.0.99 \
    ultralytics pyzmq numpy PyYAML onnx onnxslim onnxruntime-gpu onnxconverter-common \
    --break-system-packages

COPY inference_engine/ ./inference_engine/

# PASCAL_COMPAT selects the TensorRT compilation model at runtime. It is an
# image-level fact, not a user preference: it must match the TRT pinned above.
#   0 = TRT 11 strongly-typed networks; precision comes from the ONNX graph.
#   1 = TRT 8 weak typing; precision comes from builder flags. (see arch/sm_61)
ENV ZMQ_ENDPOINT=ipc:///run/zmq/detector.sock \
    MODEL_DIR=/models \
    DEVICE=cuda:0 \
    YOLO_CONFIG_DIR=/tmp \
    PASCAL_COMPAT=0

CMD ["python3", "-m", "inference_engine"]
