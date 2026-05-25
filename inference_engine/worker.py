import signal
import sys

from .engines import create_engine
from .batch import run_batch_worker
from . import stats as stats_module


def run_worker(idx, cfg, shared_name, worker_socket):
    """Entry point for each spawned worker process."""
    def _shutdown(sig, frame):
        sys.exit(0)
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGUSR1, lambda sig, frame: stats_module.log_all())

    device = f"cuda:{idx}" if cfg.device == "auto" else cfg.device
    engine = create_engine(
        cfg.engine_type,
        device         = device,
        model_dir      = cfg.model_dir,
        max_dets       = cfg.max_dets,
        precision      = cfg.precision,
        optimize       = cfg.optimize,
        max_batch_size = cfg.max_batch_size,
    )
    run_batch_worker(
        endpoint          = f"ipc://{worker_socket}",
        engine            = engine,
        max_batch_size    = cfg.max_batch_size,
        connect           = True,
        ctx               = None,
        name              = f"worker-{idx}",
        shared_model_name = shared_name,
    )
