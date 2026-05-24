FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ── System packages ────────────────────────────────────────────────────────
# libcudnn8:      CuDNN 8.x — TRT 8.6 requires CuDNN 8, not 9.
# libcublas-12-2: pinned; newer versions caused runtime failures on Pascal.
# libnccl2:       pinned alongside cublas at last known-good versions.
RUN apt-get update && apt-get install -y --no-install-recommends --allow-downgrades \
        python3 python3-pip \
        libgl1 libglib2.0-0 libgomp1 \
        libpng16-16 libjpeg-turbo8 \
        cuda-cupti-12-2=12.2.142-1 \
        libcudnn8=8.9.7.29-1+cuda12.2 \
        libcublas-12-2=12.2.5.6-1 \
        libnccl2=2.19.3-1+cuda12.2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Custom PyTorch / Torchvision wheels ────────────────────────────────────
# Official PyTorch wheels do not ship sm_61 native code. These are built from
# the v2.5.1 / v0.20.1 tags with TORCH_CUDA_ARCH_LIST=6.1 by build.sh.
COPY pytorch-workspace/src/torch-*.whl \
     pytorch-workspace/src/torchvision-*.whl \
     /tmp/wheels/
RUN pip3 install --no-cache-dir /tmp/wheels/*.whl && rm -rf /tmp/wheels

# ── Python dependencies ────────────────────────────────────────────────────
# Single pip invocation so the full dependency graph is resolved at once.
# numpy==1.26.4 is listed first in requirements.txt; pip cannot upgrade it
# past that pin without raising a conflict error.
COPY arch/sm_61/requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt
# opencv 4.13 added numpy>=2 to its PyPI metadata but works fine with 1.26.4.
# Install with --no-deps to bypass that spurious constraint.
RUN pip3 install --no-cache-dir --no-deps opencv-python==4.13.0.92

# ── TensorRT libs ─────────────────────────────────────────────────────────
# tensorrt-libs==8.6.1 was removed from PyPI (only TRT 9.x available now,
# which dropped Pascal/sm_61 support). Copy the .so bundle from the previous
# working image so the PyPI tensorrt-bindings package can find them at runtime.
COPY --from=frigate-inference:sm_61 \
     /usr/local/lib/python3.10/dist-packages/tensorrt_libs \
     /usr/local/lib/python3.10/dist-packages/tensorrt_libs
COPY --from=frigate-inference:sm_61 \
     /usr/local/lib/python3.10/dist-packages/tensorrt_libs-8.6.1.dist-info \
     /usr/local/lib/python3.10/dist-packages/tensorrt_libs-8.6.1.dist-info

# ── cu11 shared libraries ──────────────────────────────────────────────────
# onnxruntime-gpu 1.18.1 depends on CUDA 11 libs that pip installs under
# site-packages. Register them with ldconfig so the dynamic linker finds them.
RUN printf '%s\n' \
        '/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib' \
        '/usr/local/lib/python3.10/dist-packages/nvidia/cuda_runtime/lib' \
        '/usr/local/lib/python3.10/dist-packages/nvidia/cufft/lib' \
        '/usr/local/lib/python3.10/dist-packages/nvidia/curand/lib' \
        '/usr/local/lib/python3.10/dist-packages/nvidia/cusparse/lib' \
        > /etc/ld.so.conf.d/nvidia-pip-cu11.conf \
    && ldconfig

# ── Inference engine ───────────────────────────────────────────────────────
COPY inference_engine/ ./inference_engine/

ENV ZMQ_ENDPOINT=ipc:///run/zmq/detector.sock \
    MODEL_DIR=/models \
    DEVICE=cuda:0 \
    YOLO_CONFIG_DIR=/tmp

CMD ["python3", "-m", "inference_engine"]
