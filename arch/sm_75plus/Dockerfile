FROM nvidia/cuda:12-runtime-ubuntu24.04

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
    --index-url https://download.pytorch.org/whl/cu128

RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    pip3 install \
    tensorrt ultralytics pyzmq numpy PyYAML onnx onnxslim onnxruntime-gpu

COPY inference_engine/ ./inference_engine/

ENV ZMQ_ENDPOINT=ipc:///run/zmq/detector.sock \
    MODEL_DIR=/models \
    DEVICE=cuda:0 \
    YOLO_CONFIG_DIR=/tmp

CMD ["python3", "-m", "inference_engine"]
