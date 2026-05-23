#!/usr/bin/env python3
import logging
import multiprocessing
import os
import signal
import sys
import time


from .config import ServerConfig
from .engines import create_engine
from .broker import run_broker
from .batch import run_batch_worker
from .worker import run_worker
from . import stats as stats_module

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("inference")



def main():
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass  # already set
    config = ServerConfig()
    logger.info(f"Inference engine starting: {config.summary()}")

    def _shutdown(sig, frame):
        logger.info("Shutting down...")
        sys.exit(0)
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT,  _shutdown)

    # SIGUSR1: dump rolling stats to the log on demand.
    # Single-worker: handler fires in the same process as the worker — works directly.
    # Multi-worker: each spawned worker handles its own SIGUSR1; the main process
    # just forwards the signal to each worker pid.
    # Usage: docker kill --signal=SIGUSR1 frigate-inference
    worker_processes = []  # populated below in multi-worker path

    def _sigusr1(sig, frame):
        if worker_processes:
            for p in worker_processes:
                if p.is_alive():
                    os.kill(p.pid, signal.SIGUSR1)
        else:
            stats_module.log_all()

    signal.signal(signal.SIGUSR1, _sigusr1)

    if config.num_workers == 1:
        # ── Single worker: no broker overhead ────────────────────────────────
        # The batch worker binds the ROUTER socket directly. Simple and optimal
        # for a single GPU — there is no routing layer to traverse.
        engine = create_engine(
            config.engine_type,
            device    = config.device,
            model_dir = config.model_dir,
            max_dets  = config.max_dets,
            precision = config.precision,
        )
        run_batch_worker(
            endpoint      = config.endpoint,
            engine        = engine,
            max_batch_size= config.max_batch_size,
            connect       = False,
            name          = "worker",
        )

    else:
        # ── Multi-worker: broker + N batch workers ────────────────────────────
        # Clean up any stale worker socket
        worker_socket = "/tmp/workers.sock"
        try:
            os.unlink(worker_socket)
        except OSError:
            pass

        # We use a multiprocessing Manager to share the active model name
        # across all separate worker processes.
        manager = multiprocessing.Manager()
        shared_model_name = manager.Value(str, "")

        # Start broker in a separate background process
        broker_process = multiprocessing.Process(
            target=run_broker,
            kwargs=dict(
                frontend_endpoint = config.endpoint,
                backend_endpoint  = f"ipc://{worker_socket}",
            ),
            daemon=True,
            name="broker",
        )
        broker_process.start()

        # Brief sleep to allow broker to bind the IPC sockets
        time.sleep(0.5)
        logger.info(f"Broker ready. Starting {config.num_workers} batch worker processes.")

        for i in range(config.num_workers):
            p = multiprocessing.Process(
                target=run_worker,
                args=(i, config, shared_model_name, worker_socket),
                daemon=True,
                name=f"worker-{i}",
            )
            p.start()
            worker_processes.append(p)

        # Wait for worker processes to complete (they run indefinitely unless killed)
        for p in worker_processes:
            p.join()


if __name__ == "__main__":
    main()
