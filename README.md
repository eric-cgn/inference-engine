# Frigate-Compatible ZMQ Inference Pipeliner

A GPU-accelerated inference server that leverages Frigate NVR's built-in ZMQ detector
support to provide pipelined, dynamically-batching inference, with support for Pascal GPUs.

Supports multiple GPU (theoretically), multiple workers per GPU, and dynamic batch sizes.

This is not part of any official distribution, is not endorsed by anyone, and comes with
no guarantee of fitness for any purpose. Getting a working sm_61 build together was enough
of a PITA that it seemed worth sharing — and these cards are still plenty capable of
running YOLO `n` and `s` models at useful framerates.

Oh, and I should mention, it was written entirely by Claude and Gemini with very little oversight
other than a little bit of architectural guidance and some copy-pasting for testing. It
does work though, and got me off custom Frigate container builds which is very nice QOL
upgrade. Maybe you find it useful if you're the type that likes keeping old tech doing
meaningful work.

## Why this exists

**Pascal GPU support.** And for that, it's rather over-engineered. Official PyTorch 2.x wheels do not include
native code for Pascal GPUs (GTX 1050 Ti, 1060, 1070, 1080 Ti — compute capability sm_6.1).
This project ships a build pipeline that compiles PyTorch 2.5.1 from source against CUDA 12.2
for sm_6.1, bringing YOLO26 and Frigate+ models to hardware that would otherwise be left
behind. Turing and newer (RTX 2060+) work with standard wheels and need no special build.

## Models

**YOLO26n (default)** — free, auto-downloads on first use (imports Ultralytics). The latest generation model
with meaningfully better accuracy than YOLO11 at similar speed. With a TensorRT engine
compiled for your GPU, yolo26n runs at ~80 FPS on a GTX 1050 Ti** — more than enough
headroom for a significant number of cameras at 5 fps detection rates.

**Frigate+ models** — if you have a Frigate+ subscription, point your Frigate config at
your model and Frigate transfers it to the inference engine automatically over ZMQ on first
run. No manual file placement needed. See
[config/frigate-detector.yaml](config/frigate-detector.yaml) for the Frigate config snippet.

Any other YOLO-format model that ultralytics can load (`.pt`, `.onnx`) also works.

## TensorRT optimization

Compiling to a TensorRT `.engine` file gives a significant speedup (2x on Pascal) because
TRT generates GPU-native code at compile time rather than interpreting the model graph
at runtime.

With `optimize: always` in `inference.yaml` (the default), the server compiles the engine
automatically on first use — no manual step required. Compilation blocks inference for
a few minutes on first run; subsequent starts use the cached engine immediately. A
per-engine file lock ensures only one worker compiles even when `num_workers > 1`.

To pre-compile before starting the server, or to run a throughput benchmark:

```bash
./tools/run-optimize.sh yolo26n           # compile (if needed) + benchmark
./tools/run-optimize.sh yolo26n --test-only  # benchmark only, no compilation
```

`run-optimize.sh` spins up a fresh inference container, waits for the ZMQ socket, runs
`optimize.py` inside a second container against the same volumes, then tears both down.
Server logs are saved to `tools/server_last_run.log`.

Override the image tag via the `INFERENCE_IMAGE` environment variable if needed:

```bash
INFERENCE_IMAGE=frigate-inference:sm_75plus ./tools/run-optimize.sh yolo26n
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

Untested. YMMV. Standard PyTorch wheels work fine on Turing+, so the pipelining
here may or may not improve your setup over the default Frigate detector.

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

With `optimize: always` (default), the engine compiles automatically on first use.
Watch the logs for compilation progress: `docker logs -f frigate-inference`.
The first request blocks until compilation finishes (typically 1-5 minutes).
Subsequent starts load the cached `.engine` file immediately.

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
| `max_batch_size` | `1` | Maximum frames per GPU batch |
| `optimize` | `always` | `always` — auto-compile `.engine` on first use; `if_present` — use engine if found, else fall back; `never` — always load the model as named |

## Frigate configuration

See [config/frigate-detector.yaml](config/frigate-detector.yaml) for the detector and model
stanzas to add to your `config.yml`. The short version:

```yaml
detectors:
  zmq0:
    type: zmq
    endpoint: ipc:///run/zmq/detector.sock
  # zmq1:
  #   type: zmq
  #   endpoint: ipc:///run/zmq/detector.sock

model:
  path: yolo26n.engine
  labelmap_path: /config/coco.labels
  model_type: yolo-generic
  input_tensor: nhwc
  input_pixel_format: rgb
  width: 640
  height: 640
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

```
Frigate cameras
      │  (multiple REQ sockets, one per detector entry)
      ▼
  ZMQ ROUTER socket  ←── frigate-inference container
      │
  [broker]  (ROUTER → DEALER, single or multi-worker)
      │
  [batch worker(s)]
      │  phase 1: collect frames into a batch
      │  phase 2: submit batch to GPU (background thread)
      │  phase 3: decode next batch while GPU runs
      │  phase 4: send results
      ▼
  YoloEngine (ultralytics → TensorRT)
```

Multiple detector entries in Frigate's config map to multiple REQ sockets, all hitting the
same ROUTER. The broker fans them across workers. Use at least one entry per GPU. Adding
more may improve throughput by keeping the GPU better fed, or may make no difference —
it depends on camera count, frame rate, batch size, and GPU speed. For a single GPU,
`num_workers: 1` is optimal.

## A Note on the License

FWIW, anything actually copyrightable in this project is licensed
under [AGPL-3.0](LICENSE) due to its use of
[Ultralytics](https://github.com/ultralytics/ultralytics), which has produced
some very cool models and a robust and convenient library.
