"""
Rolling-window statistics for inference workers.

Data is aggregated into per-second buckets (one bucket per second, kept for
5 minutes). record() is called once per batch cycle and does nothing but
increment a handful of counters — negligible overhead on the hot path.

Stats queries iterate at most 300 buckets and are only triggered on demand
(SIGUSR1 or ZMQ stats_request), never from the inference loop.
"""
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger("stats")

# Process-wide registry so SIGUSR1 can dump all workers in one shot.
_registry: list["RollingStats"] = []
_registry_lock = threading.Lock()


@dataclass
class _Bucket:
    ts: float           # unix timestamp of this second (floor)
    frames: int = 0
    batches: int = 0
    lat_sum: float = 0.0
    lat_min: float = float("inf")
    lat_max: float = 0.0
    idle_ms: float = 0.0  # wall-ms spent waiting for work this second


class RollingStats:
    """Per-worker rolling statistics with 10s / 1m / 5m windows."""

    def __init__(self, name: str, max_seconds: int = 300):
        self.name = name
        self._buckets: deque[_Bucket] = deque(maxlen=max_seconds)
        self._current: _Bucket | None = None
        self._lock = threading.Lock()
        with _registry_lock:
            _registry.append(self)

    def record(self, batch_size: int, latency_ms: float, idle_ms: float = 0.0) -> None:
        """
        Record one completed batch cycle.
        latency_ms is the wall time from inference submission to last response sent.
        idle_ms is the wall time spent blocking for the first frame of this batch.
        Called once per batch — not per frame.
        """
        bucket_ts = int(time.time())
        with self._lock:
            if self._current is None or int(self._current.ts) != bucket_ts:
                if self._current is not None:
                    self._buckets.append(self._current)
                self._current = _Bucket(ts=float(bucket_ts))
            b = self._current
            b.frames   += batch_size
            b.batches  += 1
            b.lat_sum  += latency_ms
            b.lat_min   = min(b.lat_min, latency_ms)
            b.lat_max   = max(b.lat_max, latency_ms)
            b.idle_ms  += idle_ms

    def _window(self, buckets: list[_Bucket], now: float, seconds: int) -> dict | None:
        cutoff = now - seconds
        w = [b for b in buckets if b.ts >= cutoff]
        if not w:
            return None
        total_frames   = sum(b.frames   for b in w)
        total_batches  = sum(b.batches  for b in w)
        total_idle_ms  = sum(b.idle_ms  for b in w)
        lat_sum        = sum(b.lat_sum  for b in w)
        lat_min        = min(b.lat_min  for b in w)
        lat_max        = max(b.lat_max  for b in w)
        window_ms      = seconds * 1000
        return {
            "fps":              round(total_frames  / seconds,          2),
            "latency_avg_ms":   round(lat_sum / total_batches,          2) if total_batches else 0,
            "latency_min_ms":   round(lat_min,                          2) if lat_min != float("inf") else 0,
            "latency_max_ms":   round(lat_max,                          2),
            "avg_batch_size":   round(total_frames  / total_batches,    2) if total_batches else 0,
            "batches_per_sec":  round(total_batches / seconds,          2),
            "idle_pct":         round(total_idle_ms / window_ms * 100,  1),
        }

    def summary(self) -> dict:
        """Return a JSON-serialisable summary for all three windows."""
        now = time.time()
        with self._lock:
            buckets = list(self._buckets)
            if self._current is not None:
                buckets.append(self._current)
        return {
            "worker": self.name,
            "10s": self._window(buckets, now, 10),
            "1m":  self._window(buckets, now, 60),
            "5m":  self._window(buckets, now, 300),
        }

    def log_summary(self) -> None:
        s = self.summary()

        def fmt(w: dict | None) -> str:
            if w is None:
                return "no data yet"
            return (
                f"fps={w['fps']:.1f}  "
                f"lat={w['latency_avg_ms']:.1f}ms "
                f"[{w['latency_min_ms']:.1f}-{w['latency_max_ms']:.1f}]  "
                f"batch={w['avg_batch_size']:.1f}  "
                f"calls/s={w['batches_per_sec']:.1f}  "
                f"idle={w['idle_pct']:.1f}%"
            )

        logger.info(
            f"STATS [{s['worker']}] | "
            f"10s: {fmt(s['10s'])} | "
            f"1m: {fmt(s['1m'])} | "
            f"5m: {fmt(s['5m'])}"
        )


def log_all() -> None:
    """Dump stats for every registered worker. Called from SIGUSR1 handler."""
    with _registry_lock:
        workers = list(_registry)
    if not workers:
        logger.info("STATS: no workers registered yet")
        return
    for w in workers:
        w.log_summary()
