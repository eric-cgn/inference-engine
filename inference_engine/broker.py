import logging
import threading
import zmq

logger = logging.getLogger("broker")


def run_broker(frontend_endpoint: str, backend_endpoint: str,
               ctx: zmq.Context = None, ready: threading.Event = None):
    """
    ZMQ ROUTER-DEALER broker.

    Binds a ROUTER socket facing Frigate (frontend) and a DEALER socket
    facing batch workers (backend), then transparently proxies between them.

    When all workers run in the same container, set backend_endpoint to
    an inproc:// address and pass the shared zmq.Context — messages between
    the broker and workers are zero-copy shared-memory transfers.

    Args:
        frontend_endpoint: Frigate-facing endpoint (e.g. ipc:///run/zmq/detector.sock)
        backend_endpoint:  Worker-facing endpoint  (e.g. inproc://workers)
        ctx:               Shared zmq.Context. Required for inproc://. If None,
                           a new context is created (only suitable for tcp://).
        ready:             Optional threading.Event signalled after both sockets
                           are bound, so workers know it is safe to connect.
    """
    own_ctx = ctx is None
    if own_ctx:
        ctx = zmq.Context()

    frontend = ctx.socket(zmq.ROUTER)
    frontend.bind(frontend_endpoint)

    backend = ctx.socket(zmq.DEALER)
    backend.bind(backend_endpoint)

    logger.info(
        f"Broker started — frontend: {frontend_endpoint}  backend: {backend_endpoint}"
    )

    if ready is not None:
        ready.set()

    try:
        zmq.proxy(frontend, backend)
    except zmq.error.ContextTerminated:
        pass
    finally:
        frontend.close()
        backend.close()
        if own_ctx:
            ctx.term()
