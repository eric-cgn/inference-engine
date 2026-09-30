# Changelog

## v1.2 (unreleased)

- **TensorRT 11 support** (#3, @felalex) — TRT 11 makes strongly-typed networks
  mandatory and removed the per-precision `BuilderFlag`s, so precision must now come
  from the ONNX graph. The model is lowered to FP16 with `keep_io_types=True` before
  compilation, leaving FP32 I/O so the preprocessing path is unchanged.
- **Normalization follows the wire dtype** (#3, @felalex) — `_run_trt` divided by 255
  unconditionally, so a frame Frigate had already normalised to 0-1 was divided a
  second time and reached the model 255x too dark. Only HWC frames were affected;
  `input_tensor: nchw` masked it by converting back to uint8 first. Frigate already
  carries the dtype in the ZMQ header, so the decision no longer inspects pixel values
  and one per-frame GPU sync is dropped.
- **`PASCAL_COMPAT` selects the TRT compilation model** — `inference_engine/` is shared
  by both images, but they ship different TensorRT majors and the two APIs are not
  compatible. The active path is now declared per-image in the Dockerfile rather than
  inferred at each call site, and validated against the installed TensorRT at startup
  so a mismatch fails immediately and by name.
  - `PASCAL_COMPAT=1` (sm_61, TRT 8.6.1) — `EXPLICIT_BATCH` network, precision via
    `BuilderFlag.FP16`. Also enforces fp32, since Pascal has no Tensor Cores and runs
    FP16 at 1/64 rate.
  - `PASCAL_COMPAT=0` (sm_75plus, TRT 11.3) — strongly-typed network, FP16 from the
    lowered graph.
- **Precision is enforced against the hardware** — BF16 requires Ampere (8.0); an
  RTX 2060 (7.5) cannot execute it. Unsupported precisions now downgrade with a
  warning instead of silently building an engine labelled with a precision it does
  not have.
- **TensorRT version pinned for sm_75plus** (`tensorrt==11.3.0.99`) — unpinned, a
  rebuild silently changed TRT major version, which switches the required API *and*
  invalidates every cached `.engine`.
- **Engine cache tracks the TRT version** — engines are not portable across TensorRT
  versions, but `_check_engine` validated only model hash, batch size and precision.
  A TRT upgrade returned `use`, deserialization then returned `None`, and the caller
  raised `AttributeError` with the real cause invisible. The version is now recorded
  in `.metadata`, mismatches force a recompile, and a failed deserialization reports
  the likely cause.

## v1.1 (2026-06-02)

- **CDI-based GPU pinning** — the inference container now reserves its GPU through CDI
  (`device_ids: ${INFERENCE_CDI_DEVICE}`) instead of the legacy `--gpus`/nvidia-runtime
  path. CDI records the device in the container's creation spec, so GPU access survives a
  `systemctl daemon-reload`, which could otherwise strip the device cgroup on a
  cgroup-v2 + systemd host and silently drop the GPU (NVML "Unknown Error").
- **`install.sh` CDI setup** — generates/refreshes the CDI spec and interactively prompts
  for the GPU to pin from `nvidia-ctk cdi list`, writing `INFERENCE_CDI_DEVICE` to `.env`.

## v1.0 (2026-05-30)

Performance optimizations to the batch worker and TRT inference path:

- **Pipelined batch worker** — phase 3 decodes the next batch from ZMQ while the GPU is
  running the current one, eliminating idle time between batches
- **Float32 NCHW fast path** — Frigate's native float32 NCHW frames skip the CPU
  uint8→float32 conversion entirely and go directly to the GPU
- **Pinned memory staging** — uint8/HWC frames use a persistent pinned buffer for async
  DMA (non-blocking H2D copy), allocated once at engine load
- **Zero-copy frame passing** — frames decoded from ZMQ are passed directly to the
  background inference thread as list references; the `np.array()` batch copy is eliminated
- **Precomputed response headers** — per-frame JSON response headers reduced to a bytes
  prefix/suffix splice, avoiding `json.dumps` on every result
- **Safe lazy-reload gating** — model lazy-loads are deferred (not executed) when called
  from phase 3, preventing any concurrent model reload while inference is in flight

## v0.0

Initial release.
