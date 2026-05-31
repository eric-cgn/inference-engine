# Changelog

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
