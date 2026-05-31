# Frigate-Compatible ZMQ Inference Pipeliner

**Version: [v1.0](CHANGELOG.md)**

A GPU-accelerated TensorRT inference server for Frigate NVR. Provides pipelined inference
via Frigate's built-in ZMQ detector protocol, with support for Frigate+ models and Pascal GPUs.

This is not part of any official distribution, is not endorsed by anyone, and comes with
no guarantee of fitness for any purpose. Getting a working sm_61 build together was enough
of a PITA that it seemed worth sharing — and these cards are still plenty capable of
running YOLO `n` and `s` models at useful framerates.

AI disclosure: this was written mostly by Claude and Gemini with little oversight. It
works though, was tested on 1050 and 2060 cards, and got me off custom Frigate container
builds while maintaining TRT engine performacne, which is very nice QOL upgrade.

## Why this exists

**TensorRT inference for Frigate+.** Frigate removed its native TensorRT detector on x86_64
in recent versions, and the Frigate+ model API does not list `tensorrt` as a supported detector
type — only `zmq`, `onnx`, `openvino`, `rknn`, and `rocm`. The ZMQ detector is the only
path to TRT-accelerated inference with Frigate+ models on NVIDIA hardware. This project
is the ZMQ server on the other end of that socket.

**Pascal GPU support.** The secondary motivation and origin of the project. Official
PyTorch 2.x wheels do not include native code for Pascal GPUs (GTX 1050 Ti, 1060, 1070,
1080 Ti — compute capability sm_6.1). This project ships a build pipeline that compiles
PyTorch 2.5.1 from source against CUDA 12.2 for sm_6.1, bringing YOLO26 and Frigate+
models to hardware that would otherwise be left behind. Turing and newer (RTX 2060+)
work with standard wheels and need no special build.

## Models

**YOLO26n (default)** — free, auto-downloads on first use (imports Ultralytics). The latest generation model
with meaningfully better accuracy than YOLO11 at similar speed. With a TensorRT engine
compiled for your GPU, yolo26n runs at ~80 FPS on a GTX 1050 Ti** — more than enough
headroom for a significant number of cameras at 5 fps detection rates.

**Frigate+ models** — if you have a Frigate+ subscription, point your Frigate config at
your model and Frigate transfers it to the inference engine automatically over ZMQ on first
run. No manual file placement needed. See
[config/frigate-detector.yaml](config/frigate-detector.yaml) for the Frigate config snippet.
Once transferred, run `tools/optimize.py` to compile it to a TRT engine for maximum
performance.

Any other YOLO-format model that ultralytics can load (`.pt`, `.onnx`) also works.

## TensorRT optimization

TRT compilation happens **automatically on first use** — no manual step required. When
the inference engine receives a model it hasn't compiled yet, it compiles a `.engine`
file in the background before serving any inference requests. This gives a significant
speedup (2x on Pascal) because TRT generates GPU-native code at compile time rather than
interpreting the model graph at runtime.

**Compilation takes 2–10 minutes** depending on the model and GPU. During this time the
container is running but Frigate detections will not start. This is normal — it is not
broken. Watch the log to follow progress:

```bash
docker logs -f frigate-inference
```

You will see output like:

```
INFO  yolo_engine: Compiling TRT engine for model.onnx → model.engine ...
INFO  yolo_engine: TRT engine ready — input 'images' [1, 3, 640, 640]
```

Once the second line appears, the engine is compiled and cached. All subsequent starts
load the `.engine` file directly and are fast (~1 second).

### run-optimize.sh (optional)

`tools/run-optimize.sh` is still available if you want to pre-compile an engine before
starting the full stack, or to benchmark with `--test-only`:

```bash
./tools/run-optimize.sh your-model-name
./tools/run-optimize.sh your-model-name --test-only
```

## Directory layout

This repo lives alongside your Frigate installation as a peer directory:

```
/opt/frigate/                   ← your existing Frigate install
├── compose.yaml                ← add include: pointing here (see Setup)
├── config/
│   ├── config.yml
│   └── inference.yaml          ← copied by install.sh
└── models/

/opt/inference-engine/          ← this repo
├── install.sh
├── compose.yaml                ← included by Frigate compose
├── .env                        ← your local settings (created by install.sh)
├── .env.example
├── arch/
│   ├── sm_61/                  ← Pascal build
│   └── sm_75plus/              ← Turing+ build
├── config/
│   └── inference.yaml          ← template
├── inference_engine/
└── tools/
    ├── optimize.py
    └── run-optimize.sh
```

## Setup

### Prerequisites

- Docker with NVIDIA Container Toolkit
- Frigate NVR 0.17.1
- For Pascal builds: a Linux host with `--gpus all` access during the wheel build

### 1. Install

```bash
git clone https://github.com/eric-cgn/inference-engine /opt/inference-engine
cd /opt/inference-engine
./install.sh
```

`install.sh` creates `.env` from `.env.example`, copies `inference.yaml` into your
Frigate config directory, and prints the `include:` block to add to your Frigate
`compose.yaml`.

Edit `.env` to set your paths and image tag, then add the printed `include:` block
to the top of your Frigate `compose.yaml`:

```yaml
include:
  - path: /opt/inference-engine/compose.yaml
    env_file: /opt/inference-engine/.env
```

Also add to your `frigate` service:

```yaml
    volumes:
      - zmq_ipc:/run/zmq
    depends_on: [frigate-inference]
```

### 2. Build the image

#### Turing and newer (RTX 2060, 3060, 3080, 4090, …)

Standard PyTorch wheels work fine on Turing+. See the Performance section below for
measured results on an RTX 2060.

```bash
arch/sm_75plus/build.sh
```

Set `INFERENCE_IMAGE=frigate-inference:sm_75plus` in `.env`.

#### Pascal (GTX 1050 Ti, 1060, 1070, 1080 Ti — sm_6.1)

Official PyTorch wheels don't support Pascal. Build custom wheels first:

```bash
arch/sm_61/build.sh
```

The build script clones PyTorch v2.5.1 and Torchvision v0.20.1, compiles them inside a
Docker container with `TORCH_CUDA_ARCH_LIST=6.1`, and drops the resulting `.whl` files in
`pytorch-workspace/src/`. Subsequent runs skip the compile step if wheels are already present.
The build takes 1-3 hours depending on your CPU. It is resumable — build caches are
bind-mounted so an interrupted build picks up where it left off.

Keep `precision: fp32` in `inference.yaml` for Pascal — Pascal does not have Tensor Cores.

### 3. Start

```bash
cd /opt/frigate
docker compose up -d
```

On first start, the inference engine will compile a TRT engine for your model. Watch the
log and wait for `TRT engine ready` before expecting detections to appear in Frigate:

```bash
docker logs -f frigate-inference
```

## Configuration

`install.sh` copies `config/inference.yaml` into your Frigate config directory.
Edit it there to adjust settings. All settings can also be overridden by environment
variables.

| Setting | Default | Description |
|---|---|---|
| `endpoint` | `ipc:///run/zmq/detector.sock` | ZMQ socket path — must match Frigate |
| `model_dir` | `/models` | Directory scanned for model files |
| `device` | `cuda:0` | CUDA device |
| `precision` | `fp32` | `fp32` / `fp16` / `bf16` |
| `engine_type` | `yolo` | Inference backend (only `yolo` currently) |
| `num_workers` | `1` | Parallel workers (for multi-GPU) |
| `max_batch_size` | `16` | Maximum frames per GPU batch |

## Frigate configuration

See [config/frigate-detector.yaml](config/frigate-detector.yaml) for the detector and model
stanzas to add to your `config.yml`. The short version:

```yaml
detectors:
  zmq0:
    type: zmq
    endpoint: ipc:///run/zmq/detector.sock
  # Add more entries to increase throughput — see Tuning section.
  # zmq1:
  #   type: zmq
  #   endpoint: ipc:///run/zmq/detector.sock
  # zmq2:
  #   type: zmq
  #   endpoint: ipc:///run/zmq/detector.sock

# ── Free model (yolo26n auto-downloads on first use) ──────────────────────────
model:
  path: yolo26n
  # coco.labels ships with Frigate at /config/model_cache/coco-80.labels,
  # or download from: https://github.com/nickelc/coco-labels/blob/master/coco.labels
  labelmap_path: /config/coco.labels
  model_type: yolo-generic
  input_tensor: nhwc
  input_pixel_format: rgb
  width: 640
  height: 640

# ── Frigate+ model ────────────────────────────────────────────────────────────
# Uncomment and replace with your plus:// model URL from the Frigate+ dashboard.
# Frigate transfers the model to the inference engine automatically on first run.
# No labelmap needed — Frigate+ models include their own label set.
# model:
#   path: plus://your-model-id-here
#   model_type: yolov8
#   input_tensor: nchw
#   input_pixel_format: rgb
#   width: 640
#   height: 640
```

## Compose integration

The `frigate-inference` service and shared `zmq_ipc` volume are defined in
[compose.yaml](compose.yaml) and pulled into your Frigate compose via the `include:`
directive added during setup. No manual merging required.

## Stats

Send SIGUSR1 to the container to dump rolling stats (10s / 1m / 5m windows) to the log:

```bash
docker kill --signal=SIGUSR1 frigate-inference
```

Or query stats programmatically via ZMQ:

```python
import zmq, json
ctx  = zmq.Context()
sock = ctx.socket(zmq.REQ)
sock.connect("ipc:///run/zmq/detector.sock")
sock.send_multipart([json.dumps({"stats_request": True}).encode()])
print(json.loads(sock.recv_multipart()[0]))
```

## Architecture

With `num_workers: 1` (default, recommended for a single GPU):

```
Frigate cameras
      │  (multiple REQ sockets, one per zmq detector entry)
      ▼
  ZMQ ROUTER socket  ←── frigate-inference container
      │  (direct bind, no broker)
      ▼
  batch worker
      │  phase 1: collect frames into a batch
      │  phase 2: submit batch to GPU (background thread)
      │  phase 3: decode next batch while GPU runs
      │  phase 4: send results
      ▼
  YoloEngine (ultralytics → TensorRT)
```

With `num_workers > 1` (multi-GPU), a ROUTER→DEALER broker fans frames across workers.
For a single GPU, `num_workers: 1` eliminates the broker hop entirely.

Multiple detector entries in Frigate's config (`zmq0`, `zmq1`, …) map to multiple REQ
sockets all connecting to the same ROUTER. This is distinct from `num_workers` — see
the Tuning section below.

## Tuning

### Two separate dials

- **`num_workers` in `inference.yaml`** — the number of inference worker processes.
  One per GPU. More workers on a single GPU do not help and will contend on the CUDA
  context. Keep this at `1` for a single GPU.

- **zmq detector entries in Frigate's `config.yml`** — the number of parallel ZMQ
  pipelines Frigate maintains. This is the primary throughput lever. See below.

### Understanding ZMQ latency

Each ZMQ detector entry in Frigate is a **synchronous, blocking pipeline**: it sends one
frame to the inference engine, waits for the result, then sends the next. While it is
waiting, no other frame can go through that entry. This means a single entry can only
sustain:

```
fps_per_entry = 1000 / cycle_ms
```

where the **cycle time** is the full round-trip:

```
cycle_ms = gpu_inference_ms + frigate_overhead_ms
```

`gpu_inference_ms` is the time the GPU spends on the forward pass. `frigate_overhead_ms`
is fixed at roughly **20 ms** regardless of GPU speed — it is the cost of Frigate moving
frames between its internal camera processor subprocesses, queuing them for the detector
subprocess, and dispatching results back. You cannot reduce this by changing the inference
engine; it is intrinsic to Frigate's architecture.

**The practical consequence:** a fast GPU does not automatically increase throughput. A
GPU that runs inference in 7 ms still has a ~27 ms cycle time. One ZMQ entry can only
push ~37 fps regardless of how fast the GPU is. You need multiple ZMQ entries to keep the
GPU continuously fed.

### Measuring your actual GPU time

Query the inference engine stats directly:

```bash
docker exec frigate-inference python3 -c "
import zmq, json
ctx = zmq.Context()
sock = ctx.socket(zmq.DEALER)
sock.connect('ipc:///run/zmq/detector.sock')
sock.send_multipart([b'', json.dumps({'stats_request': True}).encode()])
msg = sock.recv_multipart()
print(json.dumps(json.loads(msg[-1])['stats']['10s'], indent=2))
"
```

The `latency_avg_ms` field is pure GPU time for the inference engine. Compare it to
Frigate's `inference_speed` stat shown in the Frigate UI — the difference is the Frigate
overhead for your system.

### Calculating how many ZMQ entries you need

Size N against the GPU's maximum throughput, not your camera count or `detect_fps`
setting. Frigate submits frames as fast as detections are needed — during active motion
scenes the rate can far exceed the per-camera fps setting — so a camera-count estimate
will undersize N when it matters most.

The GPU can theoretically process `1000 / gpu_ms` frames per second if kept fully fed.
Each ZMQ entry can push at most `1000 / cycle_ms` frames per second. To keep the GPU
continuously busy:

```
max_gpu_fps   = 1000 / gpu_inference_ms
entry_fps     = 1000 / cycle_ms
N             = ceil( max_gpu_fps / entry_fps )
              = ceil( cycle_ms / gpu_inference_ms )
```

**Worked example — RTX 2060, fp16:**

```
GPU inference latency : 7.2 ms    (from stats)
Frigate overhead      : ~20 ms    (fixed)
Cycle time            : ~27 ms

GPU max throughput    : 1000 / 7.2  ≈ 139 fps
One entry capacity    : 1000 / 27   ≈  37 fps

N = ceil(139 / 37) = ceil(3.75) = 4 entries to fully saturate the GPU
```

In practice, 3–4 entries covers most single-GPU setups. More than 4–5 is rarely
beneficial and increases average latency, since frames begin queuing in the ROUTER socket
rather than being dispatched to the GPU immediately.

### Configuring entries in Frigate

Add one stanza per entry to your Frigate `config.yml`. They all connect to the same
socket — the inference engine's ROUTER handles them all:

```yaml
detectors:
  zmq0:
    type: zmq
    endpoint: ipc:///run/zmq/detector.sock
  zmq1:
    type: zmq
    endpoint: ipc:///run/zmq/detector.sock
  zmq2:
    type: zmq
    endpoint: ipc:///run/zmq/detector.sock
```

A `model:` stanza is shared across all entries — you do not need one per entry.

More than 4–5 entries is rarely beneficial and increases average latency, since frames
begin queuing in the ROUTER socket rather than being dispatched to the GPU immediately.

## Performance

### RTX 2060 — sm_75plus container

| | |
|---|---|
| **GPU** | NVIDIA GeForce RTX 2060 |
| **Driver** | 580.159.03 |
| **Container** | `frigate-inference:sm_75plus` |
| **Model** | Frigate+ 2020.0 yolo9s base, compiled to FP16 TRT engine |
| **Input** | 640×640 |
| **ZMQ detector entries** | 3 (`zmq0`, `zmq1`, `zmq2`) |
| **num_workers** | 1 |
| **max_batch** | 1 (see note below) |
| **precision** | fp16 |
| **Cameras** | 11 cameras |

**Sustained throughput (11 cameras, ~5 fps detect per camera):**

| Metric | Value |
|--------|-------|
| Throughput | ~84 fps |
| Avg GPU inference latency | 7.2 ms |
| Min / Max latency | 5.4 ms / 13.3 ms |
| Idle (waiting for frames) | ~40% |
| CPU usage | ~68% of one core |

> **Note on batch size:** `max_batch > 1` is not currently effective with Frigate+ models.
> Frigate sends one frame per ZMQ request and does not pipeline multiple frames into a
> single message, so the batch worker always receives a batch of 1. Dynamic batching
> would require Frigate to submit frames faster than the GPU can drain them, which does
> not happen in normal single-GPU operation.

The ~20 ms Frigate pipeline overhead is on top of the 7.2 ms GPU time — Frigate's own
`inference_speed` stat will read closer to 27–30 ms. See the Tuning section for the full
worked example calculating that 3 ZMQ detector entries are the right number for this setup.

## A Note on the License

FWIW, anything actually copyrightable in this project is licensed
under [AGPL-3.0](LICENSE) due to its use of
[Ultralytics](https://github.com/ultralytics/ultralytics), which has produced
some very cool models and a robust and convenient library.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).
