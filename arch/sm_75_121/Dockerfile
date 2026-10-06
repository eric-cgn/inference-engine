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

# TensorRT is pinned. Unpinned, a rebuild silently jumps TRT major version,
# which changes the builder API *and* invalidates every cached .engine, since
# engines are not portable across TRT serialization versions. Changing this pin
# requires reviewing the PASCAL_COMPAT codepaths in yolo_engine.py.
#
# The -cu12 variant is selected explicitly. The plain `tensorrt` meta-package
# resolves to -cu13, putting a CUDA 13 build on this CUDA 12.8 base next to a
# cu128 torch. That does run -- TensorRT links the CUDA driver API rather than
# the runtime, so it uses whichever runtime is present -- but CUDA 13 requires
# driver >= 580, needlessly raising the minimum driver for an image whose every
# other component is CUDA 12.
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    pip3 install \
    tensorrt-cu12==11.3.0.99 \
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
