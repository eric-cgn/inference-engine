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
Once transferred, run `tools/optimize.py` to compile it to a TRT engine for maximum
performance.

Any other YOLO-format model that ultralytics can load (`.pt`, `.onnx`) also works.

## TensorRT optimization

After the first run, compile the model to a TensorRT `.engine` file. This gives a
significant speedup (I saw 2x on Pascal) because TRT generates GPU-native code at compile time
rather than interpreting the model graph at runtime.

`run-optimize.sh` runs on the host. It spins up a fresh inference server container,
waits for the ZMQ socket, then runs `optimize.py` inside a second container against
the same volumes. Both are torn down on exit and server logs are saved to
`tools/server_last_run.log`.

```bash
./tools/run-optimize.sh yolo26n
```

For a Frigate+ model (after Frigate has transferred it on first run):

```bash
./tools/run-optimize.sh your-model-name
```

After compilation, update your Frigate config to reference the `.engine` file:

```yaml
model:
  path: yolo26n.engine
```

To benchmark without recompiling:

```bash
./tools/run-optimize.sh yolo26n --test-only
```

Override the image tag via the `INFERENCE_IMAGE` environment variable if needed:

```bash
INFERENCE_IMAGE=frigate-inference:sm_75plus ./tools/run-optimize.sh yolo26n
```

## Directory layout

While it can be run standalone with any client that is compatible with the
Frigate ZMQ inference protocol, like the included optimization engine-builder,
when used with Frigate, can also be merged into your Frigate directory so that 
`/config` and `/models` — the paths used by the containers and by 
`tools/optimize.py` on the host — resolve to the same locations.

```
your-frigate/
├── compose.yaml
├── config/
│   ├── config.yml
│   └── inference.yaml
├── models/
├── build/                  ← inference_engine/, arch/, pyproject.toml
└── tools/
    ├── optimize.py
    └── run-optimize.sh
```

## Setup

### Prerequisites

- Docker with NVIDIA Container Toolkit
- Frigate NVR 0.17.1
- For Pascal builds: a Linux host with `--gpus all` access during the wheel build

### Turing and newer (RTX 2060, 3060, 3080, 4090, …)

Untested. YMMV. At least you should not try to use the sm_61 build though. Also,
I'm pretty sure your card still works with default builds, but maybe the pipelining here would
provide an improvement, or maybe you want to run the non-trt images for your base Frigate.

### Pascal (GTX 1050 Ti, 1060, 1070, 1080 Ti — sm_6.1)

Official PyTorch wheels don't support Pascal. Build custom wheels first:

```bash
cd arch/sm_61
./build.sh
```

The build script clones PyTorch v2.5.1 and Torchvision v0.20.1, compiles them inside a
Docker container with `TORCH_CUDA_ARCH_LIST=6.1`, and drops the resulting `.whl` files in
`pytorch-workspace/src/`. Subsequent runs skip the compile step if wheels are already present.
The build takes 1-3 hours depending on your CPU. It is resumable — build caches are
bind-mounted so an interrupted build picks up where it left off.

Then bring up the container:

```bash
docker compose up -d frigate-inference
./tools/run-optimize.sh yolo26n
```

Keep `precision: fp32` for Pascal — Pascal does not have Tensor Cores.

## Configuration

Copy `config/inference.yaml` to your config directory (mounted as `/config` in the
container) and adjust as needed. All settings can also be overridden by environment
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

See [compose.yaml](compose.yaml) for the service snippet to merge into your Frigate
compose file, including the shared ZMQ volume.

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
