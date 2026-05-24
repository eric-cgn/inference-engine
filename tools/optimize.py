#!/usr/bin/env python3
"""
TensorRT engine compiler and inference benchmark for the Frigate inference engine.

Intended to be invoked via tools/run-optimize.sh, which mounts this script into a
fresh inference container alongside the running server. Can also be run directly
inside any container that has the required dependencies and volume mounts.

Usage (via wrapper):
    ./tools/run-optimize.sh <model_name> [--test-only]

<model_name> examples:
    yolo26n          — auto-downloads weights from Ultralytics (default)
    yolo26s          — larger / more accurate variant
    your-model       — any .pt or .onnx already in /models/

Reads precision and max_batch_size from /config/inference.yaml (or $CONFIG_PATH)
so the compiled engine always matches the server's runtime settings.

Steps:
    1. Locate or download the source model (.pt or .onnx)
    2. Compile a TensorRT .engine file optimised for the host GPU
    3. Signal the running server to load the new engine
    4. Run a 15-second multi-threaded benchmark and print throughput statistics
"""

import json
import os
import sys
import threading
import time
import urllib.request

import numpy as np
import yaml
import zmq
from PIL import Image

# ── Configuration ──────────────────────────────────────────────────────────────
_CONFIG_PATH = os.environ.get("CONFIG_PATH", "/config/inference.yaml")
_config = {}
if os.path.exists(_CONFIG_PATH):
    with open(_CONFIG_PATH) as f:
        _config = yaml.safe_load(f) or {}
    print(f"Loaded config from {_CONFIG_PATH}")
else:
    print(f"[WARN] Config not found at {_CONFIG_PATH} — using defaults")

PRECISION      = _config.get("precision",      "fp32")
MAX_BATCH_SIZE = int(_config.get("max_batch_size", 16))
ENDPOINT       = _config.get("endpoint", os.environ.get("ZMQ_ENDPOINT", "ipc:///run/zmq/detector.sock"))

if len(sys.argv) < 2 or (len(sys.argv) == 2 and sys.argv[1] == "--test-only"):
    print("Usage: python3 optimize.py <model_name> [--test-only]")
    sys.exit(1)

TEST_ONLY = "--test-only" in sys.argv
if TEST_ONLY:
    sys.argv.remove("--test-only")

MODEL_NAME = sys.argv[1]
for ext in (".engine", ".onnx", ".pt"):
    if MODEL_NAME.endswith(ext):
        MODEL_NAME = MODEL_NAME[: -len(ext)]

HALF = (PRECISION == "fp16")

print(f"endpoint={ENDPOINT}  model={MODEL_NAME}  precision={PRECISION}  "
      f"max_batch={MAX_BATCH_SIZE}  half={HALF}")

# ── ZMQ connection ─────────────────────────────────────────────────────────────
ctx  = zmq.Context()
sock = ctx.socket(zmq.REQ)
sock.setsockopt(zmq.RCVTIMEO, 60_000)
sock.connect(ENDPOINT)


def send_and_recv(s, msg):
    try:
        s.send_multipart(msg)
        return s.recv_multipart()
    except zmq.error.Again:
        print("\n[ERROR] Timeout waiting for ZMQ response.", file=sys.stderr)
        sys.exit(1)


# ── 1. Locate or download source model ────────────────────────────────────────
engine_file = f"/models/{MODEL_NAME}.engine"
source_file = None

if os.path.exists(f"/models/{MODEL_NAME}.onnx"):
    source_file = f"/models/{MODEL_NAME}.onnx"
elif os.path.exists(f"/models/{MODEL_NAME}"):
    aliased = f"/models/{MODEL_NAME}.onnx"
    if not os.path.exists(aliased):
        os.symlink(f"/models/{MODEL_NAME}", aliased)
    source_file = aliased
elif os.path.exists(f"/models/{MODEL_NAME}.pt"):
    source_file = f"/models/{MODEL_NAME}.pt"
else:
    source_file = f"/models/{MODEL_NAME}.pt"
    print(f"Downloading {MODEL_NAME}.pt from Ultralytics...")
    url = f"https://github.com/ultralytics/assets/releases/latest/download/{MODEL_NAME}.pt"
    urllib.request.urlretrieve(url, source_file)

# ── 2. Compile TensorRT engine ─────────────────────────────────────────────────
if TEST_ONLY:
    print(f"--test-only: skipping TRT compilation, testing {source_file}")
    engine_file = source_file
elif os.path.exists(engine_file):
    print(f"Engine already exists at {engine_file} — skipping compilation.")
else:
    print("TRT engine not found — compiling...")
    if source_file.endswith(".onnx"):
        import onnx
        import tensorrt as trt

        print(f"Compiling from ONNX (precision={PRECISION}, max_batch={MAX_BATCH_SIZE})...")
        onnx_model  = onnx.load(source_file)
        graph_input = onnx_model.graph.input[0]
        dim0        = graph_input.type.tensor_type.shape.dim[0]
        is_dynamic  = dim0.HasField("dim_param") or not dim0.HasField("dim_value")
        batch_val   = None if is_dynamic else dim0.dim_value

        if not is_dynamic:
            print(f"Static batch={batch_val} in ONNX — making batch dimension dynamic for batch={MAX_BATCH_SIZE}...")
            import copy
            model_dyn = copy.deepcopy(onnx_model)
            for t in list(model_dyn.graph.input) + list(model_dyn.graph.output):
                t.type.tensor_type.shape.dim[0].dim_param = "batch"
            dyn_path = source_file + ".dyn.onnx"
            onnx.save(model_dyn, dyn_path)
            source_file = dyn_path
            is_dynamic = True

        logger_trt = trt.Logger(trt.Logger.INFO)
        builder    = trt.Builder(logger_trt)
        network    = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, logger_trt)
        with open(source_file, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                raise RuntimeError("Failed to parse ONNX for TRT compilation.")

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * 1024 ** 3)
        if HALF:
            config.set_flag(trt.BuilderFlag.FP16)

        if is_dynamic:
            print("Dynamic batch dimension — configuring optimization profile...")
            profile = builder.create_optimization_profile()
            inp     = network.get_input(0)
            profile.set_shape(inp.name,
                              (1, 3, 640, 640),
                              (MAX_BATCH_SIZE, 3, 640, 640),
                              (MAX_BATCH_SIZE, 3, 640, 640))
            config.add_optimization_profile(profile)
        else:
            print(f"Static batch={batch_val} — compiling fixed engine...")

        engine_bytes = builder.build_serialized_network(network, config)
        if engine_bytes is None:
            raise RuntimeError("TRT engine build failed.")
        with open(engine_file, "wb") as f:
            f.write(engine_bytes)
        print(f"Engine written to {engine_file}")

    else:
        # .pt source — let ultralytics drive the TRT export
        print(f"Compiling from .pt (precision={PRECISION}, batch={MAX_BATCH_SIZE})...")
        import ultralytics
        ultralytics.utils.checks.check_requirements = lambda *a, **kw: True
        from ultralytics import YOLO
        model = YOLO(source_file)
        model.export(format="engine", half=HALF, dynamic=True,
                     batch=MAX_BATCH_SIZE, device=0)

# ── 3. Tell the server to load the engine ─────────────────────────────────────
print(f"\nSignalling server to load: {MODEL_NAME}")
resp = send_and_recv(sock, [json.dumps({"model_request": True, "model_name": MODEL_NAME}).encode()])
resp_dict = json.loads(resp[0].decode())
print(f"Server response: {resp_dict}")
if not resp_dict.get("model_loaded"):
    print("[ERROR] Server failed to load the engine.", file=sys.stderr)
    sys.exit(1)

# ── 4. Benchmark ───────────────────────────────────────────────────────────────
image_path = "/test_client_image.png"
if os.path.exists(image_path):
    print(f"\nLoading test image from {image_path}...")
    img = Image.open(image_path).convert("RGB").resize((640, 640))
else:
    print("\nNo test image found — downloading sample...")
    url = "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r:
        img = Image.open(r).convert("RGB").resize((640, 640))

tensor     = np.array(img, dtype=np.uint8)
inf_header = json.dumps({"shape": list(tensor.shape), "dtype": "uint8"}).encode()

# Single warm-up pass before spawning benchmark threads
send_and_recv(sock, [inf_header, tensor.tobytes()])
sock.close()

TEST_DURATION    = 15.0
NUM_THREADS      = MAX_BATCH_SIZE
results_lock     = threading.Lock()
total_latencies  = []
first_pass_count = 0
end_time         = time.time() + TEST_DURATION

print(f"\nBenchmarking for {TEST_DURATION:.0f}s with {NUM_THREADS} concurrent threads "
      f"(= max_batch_size, keeps GPU fully loaded)...")


def client_thread():
    global first_pass_count
    t_ctx  = zmq.Context()
    t_sock = t_ctx.socket(zmq.REQ)
    t_sock.setsockopt(zmq.RCVTIMEO, 5000)
    t_sock.connect(ENDPOINT)

    local_latencies = []
    while time.time() < end_time:
        t0 = time.time()
        try:
            t_sock.send_multipart([inf_header, tensor.tobytes()])
            resp = t_sock.recv_multipart()
        except zmq.error.Again:
            print("Thread timeout!")
            break
        local_latencies.append((time.time() - t0) * 1000)

        with results_lock:
            if first_pass_count < 3 and len(resp) > 1:
                hdr     = json.loads(resp[0].decode())
                results = np.frombuffer(resp[1], dtype=np.float32).reshape(hdr["shape"])
                valid   = results[results[:, 1] > 0.0]
                print(f"\n[verify thread {first_pass_count + 1}] "
                      f"{len(valid)} detection(s) — first 5:")
                print(valid[:5] if len(valid) else "  (none)")
                first_pass_count += 1

    t_sock.close()
    t_ctx.term()
    with results_lock:
        total_latencies.extend(local_latencies)


threads = [threading.Thread(target=client_thread) for _ in range(NUM_THREADS)]
for t in threads:
    t.start()
for t in threads:
    t.join()

if not total_latencies:
    print("Test failed — no responses received.")
    sys.exit(1)

total_frames = len(total_latencies)
avg_lat      = sum(total_latencies) / total_frames

print(f"\n--- Client ({TEST_DURATION:.0f}s, {NUM_THREADS} threads) ---")
print(f"Frames processed : {total_frames}")
print(f"Avg latency      : {avg_lat:.2f} ms")
print(f"Throughput       : {total_frames / TEST_DURATION:.1f} fps")

print("\n--- Server rolling stats ---")
stats_sock = ctx.socket(zmq.REQ)
stats_sock.setsockopt(zmq.RCVTIMEO, 5000)
stats_sock.connect(ENDPOINT)
try:
    resp  = send_and_recv(stats_sock, [json.dumps({"stats_request": True}).encode()])
    stats = json.loads(resp[0].decode()).get("stats", {})

    def fmt(w):
        if w is None:
            return "  no data"
        return (f"  fps={w['fps']:.1f}  lat={w['latency_avg_ms']:.1f}ms "
                f"[{w['latency_min_ms']:.1f}-{w['latency_max_ms']:.1f}]  "
                f"batch={w['avg_batch_size']:.1f}  calls/s={w['batches_per_sec']:.1f}  "
                f"idle={w['idle_pct']:.1f}%")

    print(f"Worker : {stats.get('worker', '?')}")
    print(f"  10s  :{fmt(stats.get('10s'))}")
    print(f"  1m   :{fmt(stats.get('1m'))}")
    print(f"  5m   :{fmt(stats.get('5m'))}")
except Exception as e:
    print(f"  [could not fetch server stats: {e}]")
finally:
    stats_sock.close()

ctx.term()
print("\nDone.")
