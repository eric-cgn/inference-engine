# Frigate-Compatible ZMQ Inference Pipeliner

**Version: [v1.2](CHANGELOG.md)**

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
│   └── sm_75_121/              ← Turing+ build
├── config/
│   └── inference.yaml          ← template
├── inference_engine/
└── tools/
    ├── optimize.py
    └── run-optimize.sh
```

## Hardware support

Two images, split by GPU generation. The split is structural, not a packaging choice:
TensorRT 11's floor is SM 7.5 and Pascal is SM 6.1, so no single TensorRT can serve both.

| Generation | SM | Consumer cards | Image | Driver range | Precisions |
|---|---|---|---|---|---|
| Maxwell | 5.0–5.2 | GTX 750 Ti, 950, 960, 970, 980, 980 Ti, TITAN X | — | — | not supported |
| **Pascal** | **6.1** | GT 1030, GTX 1050, 1050 Ti, 1060, 1070, 1070 Ti, 1080, 1080 Ti, TITAN X (Pascal), TITAN Xp | **`sm_61`** ¹ | **525 – 580.x** ² | fp32 |
| Volta | 7.0 | TITAN V only — no GeForce part shipped on Volta | — | — | not supported |
| **Turing** | **7.5** | GTX 1650, 1650 Super, 1660, 1660 Super, 1660 Ti, RTX 2060, 2060 Super, 2070, 2070 Super, 2080, 2080 Super, 2080 Ti, TITAN RTX | **`sm_75_121`** | **≥ 570** | fp32, fp16 |
| **Ampere** | **8.6** | RTX 3050, 3060, 3060 Ti, 3070, 3070 Ti, 3080, 3080 Ti, 3090, 3090 Ti | **`sm_75_121`** | **≥ 570** | fp32, fp16, bf16 |
| **Ada Lovelace** | **8.9** | RTX 4060, 4060 Ti, 4070, 4070 Super, 4070 Ti, 4070 Ti Super, 4080, 4080 Super, 4090 | **`sm_75_121`** | **≥ 570** | fp32, fp16, bf16 |
| **Blackwell** | **12.0** | RTX 5060, 5060 Ti, 5070, 5070 Ti, 5080, 5090 | **`sm_75_121`** | **≥ 570** | fp32, fp16, bf16 |
| Post-Blackwell | — | none yet — Rubin and later are datacenter-first | — ³ | — | would need a cu13 image |

Both images are CUDA 12, but on different minor versions, which is where the two driver
floors come from:

| | `sm_61` | `sm_75_121` |
|---|---|---|
| Base image | `nvidia/cuda:12.2.2` | `nvidia/cuda:12.8.1` |
| TensorRT | 8.6.1 (pinned) | 11.3.0.99 cu12 (pinned) |
| PyTorch | 2.5.1, built from source for sm_61 | 2.11.0+cu128 (stock wheels) |
| Compiled arches | `sm_61` only | `sm_75 sm_80 sm_86 sm_90 sm_100 sm_120` |
| Compilation model | Weak typing (`PASCAL_COMPAT=1`) | Strong typing (`PASCAL_COMPAT=0`) |
| Build time | 1–3 h (compiles PyTorch) | ~12 min |

¹ The `sm_61` image also carries CUDA 11 libraries (`nvidia-cublas-cu11`,
`nvidia-cuda-runtime-cu11`, and friends) because `onnxruntime-gpu 1.18.1` requires them.
They are only reachable on the Ultralytics fallback path and are never used by the
TensorRT path, but it means that image is not purely CUDA 12.

² Pascal is the only row with an *upper* bound. Driver branch 580 is the last to support
Maxwell, Pascal and Volta; 590 and later drop them. Once your host moves past 580.x, the
`sm_61` image stops working and the card is done, independently of anything in this repo.
Every other row is open-ended.

³ The image is named for the range it actually covers rather than `sm_75plus`, because
"plus" is a promise CUDA 12 cannot keep. A cu12 toolkit can only target architectures that
existed when it shipped, so Rubin and later need a separate cu13-based image — a new
`arch/` tier, not a rebuild of this one. The `121` upper bound reflects that `sm_120`
cubins run on `sm_121` by minor-version compatibility.

No consumer part has landed above `sm_121` yet — Rubin and its successors are
datacenter-first — so the bound holds for every card in the table above. It is a
statement about what CUDA 12 can target, not a guarantee about numbering: compute
capabilities are not strictly chronological, and a future consumer part could land
anywhere.

**Not supported.** Maxwell and older are below the floor of the PyTorch wheels the `sm_61`
image is built with (`TORCH_CUDA_ARCH_LIST=6.1`, no PTX, so there is no JIT fallback).
Volta falls between the two images — the `sm_61` wheels cannot run on it and TensorRT 11
excludes it — but NVIDIA never shipped a GeForce part on Volta, so no consumer card lands
there. Datacenter parts (P100, V100, A100, and the Tesla/Quadro lines generally) are out of
scope; this targets consumer GPUs.

**Precision is enforced, not assumed.** Request a precision the card cannot execute and the
engine downgrades it with a warning rather than building something mislabelled. Pascal has no
Tensor Cores and runs FP16 at 1/64 rate, so `sm_61` is pinned to fp32. BF16 hardware starts at
Ampere, so a 2060 cannot do it regardless of what `inference.yaml` says.

### Tested on

The table above is derived from NVIDIA's support matrices and the compiled architectures in
each image. These are the cards it has actually been run on:

| Card | Arch / SM | Image | TensorRT | Model used | Covered by |
|---|---|---|---|---|---|
| **GTX 1050 Ti** | Pascal 6.1 | `sm_61` | 8.6.1 | `yolo26n` 640 fp32 | image build, engine compile, full test suite, determinism, all three wire formats, throughput |
| **RTX 2060** | Turing 7.5 | `sm_75_121` | 11.3.0.99 | Frigate+ `yolov9s` 640 fp16 | the same, plus continuous use with 14 cameras |
| **RTX 3090 Ti** | Ampere 8.6 | `sm_75_121` | 11.3 | `yolo26x-obj365` 640 fp16 | contributor-reported — fp16 compile and continuous use with 4 cameras ([#3](https://github.com/eric-cgn/inference-engine/pull/3), [#5](https://github.com/eric-cgn/inference-engine/pull/5), thanks @felalex) |

The model matters as much as the card for anything throughput-related, so it is listed per
row rather than separately — a result measured with a small model does not transfer to a
large one, or the reverse.

Everything else in the matrix is inference from the support matrices, not measurement. In
particular nothing has been run on Ada or Blackwell. If you run this on a card that is not
listed, a note either way is useful — especially a driver version, since Pascal's upper
bound is real and the floor for `sm_75_121` has not been probed below 570.

## Setup

### Prerequisites

- Docker with NVIDIA Container Toolkit (>= 1.12 for CDI)
- Frigate NVR 0.17.1
- For Pascal builds: a Linux host with `--gpus all` access during the wheel build

The container reserves its GPU through [CDI](https://github.com/cncf-tags/container-device-interface)
rather than the legacy `--gpus`/nvidia-runtime path. CDI records the device in the
container's own spec, so GPU access survives a `systemctl daemon-reload` — the legacy
path can have its device cgroup stripped by a reload and silently lose the GPU until the
container is recreated. `install.sh` generates the CDI spec and pins the device for you;
re-run it after a driver upgrade (which rewrites the spec) to refresh the pin.

### 1. Install

```bash
git clone https://github.com/eric-cgn/inference-engine /opt/inference-engine
cd /opt/inference-engine
./install.sh
```

`install.sh` creates `.env` from `.env.example`, copies `inference.yaml` into your
Frigate config directory, generates the CDI spec and prompts you to pick the GPU to
pin the inference container to (written to `INFERENCE_CDI_DEVICE` in `.env`), and prints
the `include:` block to add to your Frigate `compose.yaml`.

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

#### Turing and newer — `sm_75_121`

For any card in the Turing, Ampere, Ada or Blackwell rows of the matrix above. Stock
PyTorch wheels cover sm_75 through sm_120, so there is nothing to compile.

```bash
arch/sm_75_121/build.sh
```

Set `INFERENCE_IMAGE=frigate-inference:sm_75_121` in `.env`.

#### Pascal — `sm_61`

For the Pascal row of the matrix above. Official PyTorch wheels ship no sm_61 code, so
the wheels have to be built first:

```bash
arch/sm_61/build.sh
```

The build script clones PyTorch v2.5.1 and Torchvision v0.20.1, compiles them inside a
Docker container with `TORCH_CUDA_ARCH_LIST=6.1`, and drops the resulting `.whl` files in
`pytorch-workspace/src/`. Subsequent runs skip the compile step if wheels are already present.
The build takes 1-3 hours depending on your CPU. It is resumable — build caches are
bind-mounted so an interrupted build picks up where it left off.

The resulting wheels are built with `TORCH_CUDA_ARCH_LIST=6.1` and no PTX, so they run on
sm_61 and nothing else. `precision` is forced to fp32 on this image — Pascal has no Tensor
Cores and executes FP16 at 1/64 rate — so leaving `fp16` in `inference.yaml` is harmless but
logs a warning on startup.

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
| `precision` | `fp32` | `fp32` / `fp16` / `bf16` — see below |
| `engine_type` | `yolo` | Inference backend (only `yolo` currently) |
| `num_workers` | `1` | Parallel workers (for multi-GPU) |
| `max_batch_size` | `1` | Maximum frames per GPU batch |
| `optimize` | `always` | `always` / `if_present` / `never` — engine compilation mode |

### Two independent "data type" settings

These are easy to confuse. They are unrelated and set in different places:

| | where | what it controls |
|---|---|---|
| **`precision`** | `inference.yaml` (this project) | the numeric type the **engine computes in** — fp32 / fp16 / bf16 |
| **`input_dtype`** | Frigate's `config.yml` | the type frames are **sent over the wire** as — `int` / `float` / `float_denorm` |

`input_dtype: int` is not a reduced-precision mode. The model still computes in whatever
`precision` says; `int` only means Frigate ships raw uint8 and the ÷255 happens on the GPU
instead of Frigate's CPU. See [Wire format](#wire-format) — it is worth **+17% on an RTX
2060 and +26% on a GTX 1050 Ti**, and applies to every card.

### Precision and your GPU

`precision` is validated against the card at startup and downgraded, with a warning,
if the hardware cannot execute it. It is never silently mislabelled.

| | fp32 | fp16 | bf16 | int8 |
|---|---|---|---|---|
| Pascal — GTX 10-series (sm 6.1) | ✅ | ✗ no Tensor Cores, runs at 1/64 rate | ✗ | ✗ not implemented |
| Turing — RTX 20-series (sm 7.5) | ✅ | ✅ Tensor Cores, ~2× | ✗ no BF16 hardware | ✗ not implemented |
| Ampere and newer (sm 8.0+) | ✅ | ✅ | ✅ | ✗ not implemented |

BF16 hardware starts at Ampere, so an RTX 2060 cannot run it. On the TRT 11 path
there is also no route that produces a genuinely BF16 engine, so `bf16` builds fp16
and says so rather than recording a precision the engine does not have.

**INT8 is not supported by this engine**, on any card — `precision` accepts only
`fp32`, `fp16` and `bf16`, and anything else is rejected at startup. The hardware is
capable: Turing and newer have INT8 Tensor Cores, and INT8 inference is typically
~2× fp16. What is missing is the quantization step. TensorRT 11 removed the INT8
builder flag in favour of explicit Q/DQ nodes, so using it means quantizing the ONNX
model first — a calibration pass over representative frames, with accuracy validation,
since INT8 can cost detection quality in a way fp16 does not. That is a feature, not a
flag, and it is out of scope here.

### `PASCAL_COMPAT`

Not a user setting. It selects which TensorRT API generation the compile path drives,
and is baked into each image by its Dockerfile because it has to match the TensorRT
installed there:

| Image | TensorRT | `PASCAL_COMPAT` | Compilation model |
|---|---|---|---|
| `sm_61` | 8.6.1 (pinned) | `1` | Weak typing — `EXPLICIT_BATCH` network, precision via `BuilderFlag` |
| `sm_75_121` | 11.3.0.99 (pinned) | `0` | Strong typing — precision carried by the ONNX graph |

The two APIs are mutually exclusive: TRT 11 removed the per-precision `BuilderFlag`s,
and on TRT 8 the strongly-typed call selects *implicit* batch, which the ONNX parser
rejects. Because TensorRT 11's floor is SM 7.5 and Pascal is SM 6.1, no single
TensorRT can serve both cards — the split is structural, not a workaround.

It is deliberately read only from the environment and not from `inference.yaml`. The
engine verifies it against the installed TensorRT on startup and refuses to run on a
mismatch, so a wrong value fails immediately by name rather than as an opaque ONNX
parse error during a background compile.

### `CUDA_GRAPHS`

Env-only, on by default. `CUDA_GRAPHS=0` turns it off.

`enqueueV3` launches every kernel in the engine from the CPU, one at a time. With graphs on,
that sequence is recorded once per batch size and replayed with a single call, so the
per-kernel launch cost disappears. A batch size the engine cannot accept falls back to a
normal enqueue for that size only, and graphs are discarded when an engine is reloaded.

**How much it helps depends on the model, not the card.** Launch overhead scales with the
number of kernels, so a big model has more of it to remove:

| Model | Card | Effect |
|---|---|---|
| `yolo26x-obj365` 640 fp16 | RTX 3090 Ti | ~7.2 ms → 5.8 ms engine latency (contributor-reported) |
| Frigate+ `yolov9s` 640 fp16 | RTX 2060 | no measurable change — the difference flipped sign between runs |
| `yolo26n` 640 fp32 | GTX 1050 Ti | no measurable change — 105.1 fps either way |

Graphs are recorded successfully on both TRT 11.3 and TRT 8.6.1, so the small-model result
is a real measurement rather than a silent fallback. Nothing has shown graphs to be *slower*,
so they are left on: a large model gains, a small one is unaffected.

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
  input_dtype: int          # see "Wire format" below — Frigate's defaults are the slow path
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
#   input_tensor: nhwc
#   input_dtype: int
#   input_pixel_format: rgb
#   width: 640
#   height: 640
```

### Wire format
<a id="wire-format"></a>

`input_tensor` and `input_dtype` decide what Frigate puts on the socket, and its
defaults (`nchw` / `float`) are the slowest combination. Set both explicitly.

With `int`, Frigate ships the raw uint8 frame — 1.23 MB at 640×640×3 — and the ÷255
happens on the GPU. With `float` it converts to float32 and divides on the CPU first,
putting 4.92 MB on the socket for the same pixels. Measured with Frigate out of the
loop, four concurrent clients against one engine:

| | `int` + `nhwc` | `float` + `nchw` | |
|---|---|---|---|
| RTX 2060 (TRT 11.3) | **23.0 ms** / 173 fps | 27.6 ms / 144 fps | +17% |
| GTX 1050 Ti (TRT 8.6.1) | **38.4 ms** / 104 fps | 48.6 ms / 82 fps | +26% |

At a single client the gap is wider — +25% on the 2060, +42% on the 1050 — because the
per-frame transport cost is a larger share of the total. All three wire formats produce
identical detections; only the cost differs.

**Frigate+ models ignore these settings.** Their metadata carries `inputShape` and
`inputDataType`, and Frigate applies those after your config, so neither the `model:`
block nor a per-detector `model:` override changes the resolved wire format. A `plus://`
model uses whatever it was published with — commonly `nchw` + `float`. That path is not
slow: it is exactly what the float32 NCHW fast path exists to accelerate, uploading the
frame to the GPU with no CPU conversion at all. The table above applies to local models
(`yolo26n`, your own `.onnx`), where the setting is yours to choose.

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

### RTX 2060 — sm_75_121 container

| | |
|---|---|
| **GPU** | NVIDIA GeForce RTX 2060 |
| **Driver** | 580.159.03 |
| **Container** | `frigate-inference:sm_75_121` |
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
