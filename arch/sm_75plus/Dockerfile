FROM nvidia/cuda:12-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip \
        libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Update --index-url when PyTorch publishes wheels for a newer CUDA version.
RUN pip3 install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128

RUN pip3 install --no-cache-dir \
    tensorrt ultralytics pyzmq numpy PyYAML onnx onnxslim onnxruntime-gpu

COPY inference_engine/ ./inference_engine/

ENV ZMQ_ENDPOINT=ipc:///run/zmq/detector.sock \
    MODEL_DIR=/models \
    DEVICE=cuda:0 \
    YOLO_CONFIG_DIR=/tmp

CMD ["python3", "-m", "inference_engine"]
