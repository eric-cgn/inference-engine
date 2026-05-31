import hashlib
import json
import logging
import os
import socket as _socket
import time
import threading
import numpy as np
import zmq
from .engine import InferenceEngine
from .stats import RollingStats

logger = logging.getLogger("batch")


# Global model name tracking for multi-worker environments
_CURRENT_MODEL_NAME = None
_CURRENT_MODEL_LOCK = threading.Lock()


def _base(name: str) -> str:
    """Strip model file extension for comparison."""
    for ext in (".engine", ".onnx", ".pt"):
        if name.endswith(ext):
            return name[:-len(ext)]
    return name


def _decode_msgs(msgs, engine, max_batch_size, stats=None, shared_model_name=None,
                 allow_lazy_load: bool = True):
    """
    Parse raw ZMQ multipart messages into control replies and inference frames.
    Returns:
        control_replies - list of (identity, bytes) to send immediately
        identities      - list of identity frames for inference responses
        request_ids     - list of request_id values (parallel to identities/frames)
        frames          - list of decoded numpy arrays ready for batching
        deferred        - messages to re-queue (model not loaded yet)
    """
    global _CURRENT_MODEL_NAME
    control_replies = []
    identities = []
    request_ids = []
    frames = []
    deferred = []

    for msg in msgs[:max_batch_size]:
        ident = msg[0]
        try:
            header = json.loads(msg[2].decode())
        except Exception as e:
            logger.error(f"Bad header from {ident}: {e}")
            continue

        req_id = header.get("request_id")

        if header.get("stats_request"):
            resp = json.dumps({"stats": stats.summary() if stats else {}, "request_id": req_id}).encode()
            control_replies.append((ident, resp))
            continue

        if header.get("model_request"):
            name = header.get("model_name")
            if not name:
                logger.error("model_request missing model_name")
                resp = json.dumps({"model_available": False, "model_loaded": False, "request_id": req_id}).encode()
            else:
                engine._compile_failed = False
                path = f"{engine.model_dir}/{name}"
                loaded = engine.load_model(path)
                if loaded:
                    if shared_model_name is not None:
                        shared_model_name.value = name
                    else:
                        with _CURRENT_MODEL_LOCK:
                            _CURRENT_MODEL_NAME = name
                resp = json.dumps({"model_available": loaded, "model_loaded": loaded, "request_id": req_id}).encode()
            control_replies.append((ident, resp))
            continue

        if header.get("model_data"):
            name = header.get("model_name")
            if not name or len(msg) < 4:
                logger.error("model_data missing model_name or payload frame")
                resp = json.dumps({"model_saved": False, "model_loaded": False, "request_id": req_id}).encode()
            else:
                save_name = name
                if not os.path.splitext(name)[1]:
                    save_name = name + ".onnx"

                path = f"{engine.model_dir}/{save_name}"
                try:
                    data = msg[3]
                    incoming_hash = hashlib.sha256(data).hexdigest()
                    skip_write = False
                    if os.path.exists(path):
                        h = hashlib.sha256()
                        with open(path, "rb") as f:
                            for chunk in iter(lambda: f.read(1 << 20), b""):
                                h.update(chunk)
                        skip_write = (h.hexdigest() == incoming_hash)

                    if skip_write:
                        logger.info(f"Model payload unchanged ({len(data)} bytes), skipping write: {save_name}")
                    else:
                        with open(path, "wb") as f:
                            f.write(data)
                        logger.info(f"Saved model payload to {path} ({len(data)} bytes)")

                    engine._compile_failed = False
                    loaded = engine.load_model(f"{engine.model_dir}/{name}")
                    if loaded:
                        if shared_model_name is not None:
                            shared_model_name.value = name
                        else:
                            with _CURRENT_MODEL_LOCK:
                                _CURRENT_MODEL_NAME = name
                    resp = json.dumps({"model_saved": True, "model_loaded": loaded, "request_id": req_id}).encode()
                except Exception as e:
                    logger.error(f"Failed to save model {name}: {e}")
                    resp = json.dumps({"model_saved": False, "model_loaded": False, "request_id": req_id}).encode()
            control_replies.append((ident, resp))
            continue

        if "shape" in header:
            # Check if this worker needs to lazily load the current active model
            if shared_model_name is not None:
                current_model = shared_model_name.value
            else:
                with _CURRENT_MODEL_LOCK:
                    current_model = _CURRENT_MODEL_NAME

            if current_model:
                is_loaded = (
                    engine.model is not None
                    and engine.model_name is not None
                    and _base(engine.model_name) == _base(current_model)
                )
                if (not is_loaded
                        and not getattr(engine, '_compiling', False)
                        and not getattr(engine, '_compile_failed', False)):
                    if not allow_lazy_load:
                        deferred.append(msg)
                        continue
                    logger.info(f"Worker lazily loading active model: {current_model}")
                    engine.load_model(f"{engine.model_dir}/{current_model}")

            if engine.model is None:
                rh = json.dumps({
                    "shape": [engine.max_dets, 6],
                    "dtype": "float32",
                    "request_id": req_id,
                }).encode()
                zeros = np.zeros((engine.max_dets, 6), np.float32).tobytes()
                control_replies.append((ident, rh, zeros))
            else:
                shape = tuple(header["shape"])
                dtype = np.dtype(header.get("dtype", "uint8"))
                frame = np.frombuffer(msg[3], dtype=dtype).reshape(shape)
                if frame.ndim == 4:
                    frame = frame[0]
                identities.append(ident)
                request_ids.append(req_id)
                frames.append(frame)

    return control_replies, identities, request_ids, frames, deferred


def run_batch_worker(endpoint: str, engine: InferenceEngine,
                     max_batch_size: int = 16, connect: bool = False,
                     ctx: zmq.Context = None, name: str = "worker",
                     shared_model_name=None):
    """
    Pipelined dynamic batch inference worker.

    Standalone mode (connect=False, default):
        Binds a ROUTER socket directly to `endpoint`. Use this when running
        a single worker with no broker. ctx is created internally.

    Broker mode (connect=True):
        Connects a DEALER socket to the broker's backend `endpoint`. The broker
        preserves the full message envelope, so this worker handles identities
        identically to standalone mode.

    Pipeline:
        Phase 1 - Block for a batch (or use pre-staged frames from last run).
        Phase 2 - Submit inference to a background thread (GPU starts immediately).
        Phase 3 - Select on two events while GPU runs:
                    * ZMQ socket ready  -> decode arriving frames into next batch
                    * Inference done    -> break, submit next batch immediately
        Phase 4 - Send responses.
    """
    own_ctx = ctx is None
    if own_ctx:
        ctx = zmq.Context()

    if connect:
        sock = ctx.socket(zmq.DEALER)
        sock.connect(endpoint)
        logger.info(f"Batch worker connected to broker backend: {endpoint} (max_batch={max_batch_size})")
    else:
        sock = ctx.socket(zmq.ROUTER)
        sock.bind(endpoint)
        logger.info(f"Batch worker bound to {endpoint} (standalone, max_batch={max_batch_size})")

    stats = RollingStats(name)

    total_frames = 0
    total_batches = 0

    # Staged next batch: decoded while the GPU is running the current one.
    next_identities = []
    next_request_ids = []
    next_frames = []
    deferred_msgs = []  # held when model not yet loaded

    # ── Inference pipe (socketpair) ──────────────────────────────────────────
    notify_r, notify_w = _socket.socketpair()
    notify_r.setblocking(False)

    poller = zmq.Poller()
    poller.register(sock,              zmq.POLLIN)
    poller.register(notify_r.fileno(), zmq.POLLIN)

    inference_result      = None
    inference_error       = None
    inference_identities  = []
    inference_request_ids = []

    def _run_inference_bg(tensors, idents, req_ids):
        nonlocal inference_result, inference_error
        try:
            inference_result = engine.run_inference(tensors)
            inference_error  = None
        except BaseException as e:
            inference_error  = e
            inference_result = None
        inference_identities[:] = idents
        inference_request_ids[:] = req_ids
        try:
            notify_w.send(b"\x00")
        except OSError:
            pass

    inference_thread = None

    def submit_inference(frames, idents, req_ids):
        nonlocal inference_thread, inference_result, inference_error
        inference_result = None
        inference_error  = None
        inference_thread = threading.Thread(
            target=_run_inference_bg,
            args=(frames, list(idents), list(req_ids)),
            daemon=True,
        )
        inference_thread.start()

    def drain_socket(budget):
        msgs = []
        while len(msgs) < budget:
            try:
                msgs.append(sock.recv_multipart(flags=zmq.DONTWAIT))
            except zmq.error.Again:
                break
        return msgs

    _resp_prefix = f'{{"shape": [{engine.max_dets}, 6], "dtype": "float32", "request_id": '.encode()
    _resp_suffix = b'}'

    def make_resp_header(req_id) -> bytes:
        return _resp_prefix + (str(req_id).encode() if req_id is not None else b'null') + _resp_suffix

    # ── Main loop ────────────────────────────────────────────────────────────
    while True:
        # ── PHASE 1: Acquire a batch to run ──────────────────────────────────
        idle_ms = 0.0
        if next_frames:
            run_identities  = next_identities
            run_request_ids = next_request_ids
            run_frames      = next_frames
            next_identities  = []
            next_request_ids = []
            next_frames      = []
        else:
            raw_msgs = deferred_msgs[:]
            deferred_msgs.clear()

            try:
                t_idle = time.time()
                raw_msgs.append(sock.recv_multipart())
                idle_ms = (time.time() - t_idle) * 1000
            except zmq.error.ContextTerminated:
                break

            raw_msgs.extend(drain_socket(max_batch_size - len(raw_msgs)))

            ctrl, run_identities, run_request_ids, run_frames, deferred_msgs = _decode_msgs(
                raw_msgs, engine, max_batch_size, stats, shared_model_name
            )
            for reply in ctrl:
                sock.send_multipart([reply[0], b""] + list(reply[1:]))

            if not run_frames:
                continue

        # ── PHASE 2: Start GPU inference in background ────────────────────────
        t_submit = time.time()
        batch_size = len(run_frames)
        if batch_size > 1:
            logger.debug(f"Dynamic batch: {batch_size} frames")

        submit_inference(run_frames, run_identities, run_request_ids)

        # ── PHASE 3: Select on two events while GPU is busy ──────────────────
        inference_finished = False
        while not inference_finished and len(next_frames) < max_batch_size:
            ready = dict(poller.poll())

            if notify_r.fileno() in ready:
                notify_r.recv(1)
                inference_finished = True

            if sock in ready:
                raw_next = drain_socket(max_batch_size - len(next_frames))
                if raw_next:
                    ctrl, new_idents, new_req_ids, new_frames, more_deferred = _decode_msgs(
                        raw_next, engine, max_batch_size, stats, shared_model_name,
                        allow_lazy_load=False
                    )
                    deferred_msgs.extend(more_deferred)
                    for reply in ctrl:
                        sock.send_multipart([reply[0], b""] + list(reply[1:]))
                    next_identities.extend(new_idents)
                    next_request_ids.extend(new_req_ids)
                    next_frames.extend(new_frames)

        # ── PHASE 4: Send responses ───────────────────────────────────────────
        if not inference_finished:
            inference_thread.join()
            # Drain the notification byte the thread sent but phase 3 didn't consume
            # (phase 3 exited via next-batch-full path, not via the notify signal).
            try:
                notify_r.recv(1)
            except OSError:
                pass

        if inference_error or inference_result is None:
            if inference_error:
                logger.error(f"Inference error: {inference_error}")
            else:
                logger.error("Inference hard crash (no exception): resetting engine for reload")
                engine.model = None
            for i, ident in enumerate(inference_identities):
                rh = make_resp_header(inference_request_ids[i] if i < len(inference_request_ids) else None)
                zeros = np.zeros((engine.max_dets, 6), np.float32).tobytes()
                sock.send_multipart([ident, b"", rh, zeros])
        else:
            for i, ident in enumerate(inference_identities):
                rh = make_resp_header(inference_request_ids[i] if i < len(inference_request_ids) else None)
                sock.send_multipart([ident, b"", rh, inference_result[i].tobytes()])

        # ── Stats ─────────────────────────────────────────────────────────────
        latency_ms = (time.time() - t_submit) * 1000
        stats.record(batch_size, latency_ms, idle_ms)
        total_frames  += batch_size
        total_batches += 1
