import hashlib
import json
import logging
import os
import re
import threading
import time
import numpy as np

# Frigate Plus model names are MD5 hashes (32 lowercase hex chars).
# Ultralytics model names are short alphanumeric slugs (e.g. yolo26n).
_HASH_RE = re.compile(r'^[0-9a-f]{32,}$')

logger = logging.getLogger("yolo_engine")

from ..engine import InferenceEngine


class YoloEngine(InferenceEngine):
    """
    YOLO inference engine with two execution paths:

    Direct TRT path (.engine files):
        Bypasses Ultralytics entirely. Preprocessing and decoding run in PyTorch
        on the GPU; NMS uses torchvision.ops.nms (CUDA kernel). A single
        .cpu() call collects final detections — no per-anchor GPU→CPU transfers.

    Ultralytics fallback (.onnx / .pt files):
        Used when no compiled engine is available. Slower due to Python-side
        pre/post-processing.

    The `optimize` config parameter controls engine compilation:
        always     — compile .engine on first use if not present (blocks until done)
        if_present — use .engine if found, else fall back to source model (default)
        never      — load exactly the path given, no engine substitution
    """

    # ── Shared TRT runtime (one per process) ──────────────────────────────
    _trt_runtime = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._validate_trt_profile()

    def _validate_trt_profile(self):
        """
        Verify PASCAL_COMPAT agrees with the TensorRT that is actually installed.

        PASCAL_COMPAT is declared per-image in the Dockerfile rather than sniffed
        at runtime, so the active compilation model is explicit and greppable.
        This check exists so that a mismatch fails immediately and legibly,
        instead of surfacing later as an opaque ONNX parse error inside a
        background compile thread.

        It keys on the TRT version, not on the GPU, because what the flag
        selects is which TensorRT API generation to drive -- and that is a
        property of the image, not of the card in the slot. Whether the GPU can
        execute the requested precision is a separate question, handled by
        InferenceEngine._enforce_precision.

            PASCAL_COMPAT=1  needs TRT < 11  (BuilderFlag precision still exists)
            PASCAL_COMPAT=0  needs TRT >= 10 (create_network(0) == explicit batch)

        TRT 10 satisfies both and is accepted either way.
        """
        try:
            import tensorrt as trt
        except ImportError:
            return   # Ultralytics-only fallback; no TRT path to validate

        try:
            major = int(trt.__version__.split(".")[0])
        except (ValueError, AttributeError):
            logger.warning(
                "Could not parse the installed TensorRT version — "
                "skipping PASCAL_COMPAT validation"
            )
            return

        if self.pascal_compat and major >= 11:
            raise RuntimeError(
                f"PASCAL_COMPAT=1 but TensorRT {trt.__version__} is installed. "
                f"TRT 11 removed the per-precision BuilderFlags that this path "
                f"depends on. Set PASCAL_COMPAT=0 in the Dockerfile for this image."
            )
        if not self.pascal_compat and major < 10:
            raise RuntimeError(
                f"PASCAL_COMPAT=0 but TensorRT {trt.__version__} is installed. "
                f"On TRT 8, create_network(0) selects implicit batch, which the "
                f"ONNX parser rejects. Set PASCAL_COMPAT=1 in the Dockerfile for "
                f"this image."
            )

        if self.precision == "bf16" and not self.pascal_compat:
            # onnxconverter-common lowers only to IEEE half, and TRT 11 has no
            # BF16 builder flag, so no path here produces a genuinely BF16
            # engine. Record what is actually built rather than labelling an
            # FP16 engine "bf16" -- _check_engine compares that string, so a
            # wrong label would silently never trigger a rebuild.
            logger.warning(
                "precision='bf16' has no supported TRT compile path — building fp16 instead"
            )
            self.precision = "fp16"

        logger.info(
            f"TRT profile: PASCAL_COMPAT={int(self.pascal_compat)} "
            f"(tensorrt {trt.__version__}, "
            f"{'weak typing / BuilderFlag' if self.pascal_compat else 'strongly typed / ONNX-lowered'})"
        )

    @staticmethod
    def _trt_version() -> str:
        """Installed TensorRT version, or '' when TRT is unavailable."""
        try:
            import tensorrt as trt
            return str(trt.__version__)
        except Exception:
            return ""

    @classmethod
    def _get_runtime(cls):
        if cls._trt_runtime is None:
            import tensorrt as trt
            cls._trt_runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        return cls._trt_runtime

    # ── Model loading ──────────────────────────────────────────────────────

    def load_model(self, path: str) -> bool:
        try:
            original_path = path
            ext = os.path.splitext(path)[1]
            base = path[:-len(ext)] if ext in (".onnx", ".pt", ".engine") else path
            engine_path = base + ".engine"

            if self.optimize == "never":
                resolved = path
                logger.debug(f"optimize=never — using model as given: {path}")

            elif self.optimize == "if_present":
                if os.path.exists(engine_path):
                    logger.info(f"Engine file found, using TRT path: {engine_path}")
                    resolved = engine_path
                else:
                    resolved = self._resolve_source(path)
                    if resolved is None:
                        logger.error(f"Model not found: {path}")
                        return False
                    logger.info(f"No engine at {engine_path} — loading source model: {resolved}")

            elif self.optimize == "always":
                meta_path = base + ".metadata"
                source    = self._resolve_source(path)

                status, reason = self._check_engine(source, engine_path, meta_path)

                if status == "use":
                    logger.info(f"Engine valid (metadata verified): {os.path.basename(engine_path)}")

                elif status == "write_meta":
                    logger.info(f"Engine found without metadata — writing: {os.path.basename(meta_path)}")
                    if source:
                        self._write_metadata(source, engine_path, meta_path)

                else:  # compile or recompile
                    orig_ext  = os.path.splitext(original_path)[1]
                    bare_name = not orig_ext
                    stem      = os.path.splitext(os.path.basename(original_path))[0]

                    if source is None and not bare_name:
                        logger.error(
                            f"optimize=always: no source model found for '{path}' "
                            f"— place a .onnx or .pt file in {self.model_dir}"
                        )
                        return False

                    if source is None and bare_name and _HASH_RE.match(stem):
                        # Hash-named model with no local file — Frigate will send the
                        # bytes via model_data. Signal False so Frigate sends the data.
                        return False

                    if self._compiling:
                        logger.info(f"Compilation in progress: {os.path.basename(engine_path)}")
                        return True

                    self._compiling = True
                    self.model_name = os.path.basename(original_path)

                    _src    = source
                    _status = status
                    _reason = reason
                    _bare   = bare_name
                    _stem   = stem

                    def _bg():
                        try:
                            src = _src
                            if src is None and _bare:
                                src = self._try_ultralytics_download(_stem)
                                if src is None:
                                    logger.error(f"Auto-download failed for '{_stem}' — compilation aborted")
                                    self._compile_failed = True
                                    return
                            verb = "Compiling" if _status == "compile" else "Recompiling"
                            logger.info(f"{verb} engine (background): {_reason}")
                            self._compile_engine_locked(src, engine_path, meta_path)
                            self._load_trt_direct(engine_path)
                            self.model_path  = engine_path
                            self.model_mtime = os.path.getmtime(engine_path)
                            self._compile_failed = False
                            logger.info(f"Background compilation complete — engine ready: {os.path.basename(engine_path)}")
                        except Exception as e:
                            logger.error(f"Background compilation failed: {e}")
                            self._compile_failed = True
                        finally:
                            self._compiling = False

                    threading.Thread(target=_bg, daemon=True).start()
                    return True

                resolved = engine_path

            else:
                logger.warning(f"Unknown optimize='{self.optimize}', falling back to if_present")
                resolved = engine_path if os.path.exists(engine_path) else self._resolve_source(path) or path

            mtime = os.path.getmtime(resolved) if os.path.exists(resolved) else 0.0
            if (self.model is not None and self.model_path == resolved
                    and self.model_mtime == mtime):
                logger.debug(f"Model {os.path.basename(original_path)} already loaded (up to date).")
                return True

            logger.info(f"Loading YOLO model: {resolved}  device={self.device}  precision={self.precision}")

            if resolved.endswith(".engine"):
                self._load_trt_direct(resolved)
            else:
                self._load_ultralytics(resolved)

            self.model_name  = os.path.basename(original_path)
            self.model_path  = resolved
            self.model_mtime = mtime
            logger.info(f"Model {self.model_name} loaded.")
            return True
        except Exception as e:
            logger.error(f"Failed to load model {path}: {e}")
            return False

    def _resolve_source(self, path: str):
        """Return a loadable source path (.onnx or .pt), or None if not found."""
        ext = os.path.splitext(path)[1]
        if ext in (".onnx", ".pt") and os.path.exists(path):
            return path
        if not ext and os.path.exists(path):
            # Extension-less file — Frigate stores ONNX without extension
            alias = path + ".onnx"
            if not os.path.exists(alias):
                os.symlink(os.path.basename(path), alias)
            return alias
        for candidate_ext in (".onnx", ".pt"):
            candidate = path + candidate_ext
            if os.path.exists(candidate):
                return candidate
        return None

    def _try_ultralytics_download(self, stem: str) -> str | None:
        """Download a named ultralytics model .pt to model_dir. Returns path or None."""
        try:
            import ultralytics
            ultralytics.utils.checks.check_requirements = lambda *a, **kw: True
            from ultralytics import YOLO
            target = os.path.join(self.model_dir, stem + ".pt")
            logger.info(f"Downloading {stem}.pt via ultralytics → {target}")
            YOLO(target)
            if os.path.exists(target):
                logger.info(f"Download complete: {target}")
                return target
            logger.error(f"ultralytics download did not produce {target}")
            return None
        except Exception as e:
            logger.error(f"ultralytics download failed for '{stem}': {e}")
            return None

    @staticmethod
    def _sha256(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return "sha256:" + h.hexdigest()

    def _check_engine(self, source_path, engine_path: str, meta_path: str):
        """
        Decide whether the engine is usable as-is or needs action.

        Returns (status, reason):
          "use"        — hashes + batch + precision all match, proceed
          "compile"    — engine file does not exist
          "recompile"  — engine exists but is stale (reason explains why)
          "write_meta" — engine exists, metadata absent (pre-compiled / old install)
        """
        if not os.path.exists(engine_path):
            return ("compile", "engine file not found")

        if not os.path.exists(meta_path):
            return ("write_meta", "metadata missing")

        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except Exception as e:
            return ("recompile", f"metadata unreadable: {e}")

        if meta.get("engine_batch") != self.max_batch_size:
            return ("recompile",
                    f"batch size changed {meta.get('engine_batch')} → {self.max_batch_size}")

        if meta.get("precision") != self.precision:
            return ("recompile",
                    f"precision changed {meta.get('precision')} → {self.precision}")

        # A serialized engine can only be deserialized by the TRT version that
        # built it. Without this check a TRT upgrade yields "use", and the
        # deserialize then fails at load time with detection already offline.
        #
        # Only act on it when there is a source model to rebuild from. For an
        # engine-only deployment a recompile is impossible, and returning
        # "recompile" there would turn a working setup into a hard failure in
        # load_model; warn and let _load_trt_direct report it if it really is
        # incompatible.
        stored_trt  = meta.get("trt_version", "")
        current_trt = self._trt_version()
        can_rebuild = bool(source_path and os.path.exists(source_path))
        if can_rebuild:
            if not stored_trt:
                return ("recompile",
                        "engine predates TRT version tracking — rebuilding to "
                        "guarantee it matches the installed TensorRT")
            if current_trt and stored_trt != current_trt:
                return ("recompile",
                        f"TensorRT changed {stored_trt} → {current_trt} "
                        f"(engines are not portable across TRT versions)")
        elif stored_trt and current_trt and stored_trt != current_trt:
            logger.warning(
                f"Engine was built by TensorRT {stored_trt} but {current_trt} is "
                f"installed, and no source model is available to rebuild from — "
                f"loading it will probably fail."
            )

        if source_path and os.path.exists(source_path):
            current = self._sha256(source_path)
            stored  = meta.get("model_hash", "")
            if current != stored:
                return ("recompile",
                        f"source model changed ({stored[:15]}… → {current[:15]}…)")

        current_eng = self._sha256(engine_path)
        if current_eng != meta.get("engine_hash", ""):
            return ("recompile", "engine file modified externally")

        return ("use", None)

    def _write_metadata(self, source_path: str, engine_path: str, meta_path: str):
        meta = {
            "model":        os.path.basename(source_path) if source_path else None,
            "model_hash":   self._sha256(source_path) if source_path and os.path.exists(source_path) else None,
            "engine":       os.path.basename(engine_path),
            "engine_hash":  self._sha256(engine_path),
            "engine_batch": self.max_batch_size,
            "precision":    self.precision,
            "engine_type":  "yolo",
            # Serialized engines are not portable across TRT versions; record
            # which one built this so _check_engine can force a rebuild.
            "trt_version":  self._trt_version(),
        }
        tmp = meta_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(meta, f, indent=2)
        try:
            os.rename(tmp, meta_path)
        except FileNotFoundError:
            if not os.path.exists(meta_path):
                raise
        logger.info(f"Metadata written: {os.path.basename(meta_path)}")

    def _compile_engine_locked(self, source_path: str, engine_path: str,
                                meta_path: str = None):
        """Compile source_path → engine_path, serialized across workers via flock."""
        import fcntl
        lock_path = engine_path + ".lock"
        with open(lock_path, "w") as lf:
            waited = False
            try:
                fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                logger.info(f"Another worker is compiling — waiting: {lock_path}")
                fcntl.flock(lf, fcntl.LOCK_EX)
                waited = True

            if waited:
                # Re-validate: another worker may have just finished a good engine
                if meta_path:
                    status, _ = self._check_engine(source_path, engine_path, meta_path)
                    if status == "use":
                        logger.info(f"Engine compiled and validated by another worker: {os.path.basename(engine_path)}")
                        return
                elif os.path.exists(engine_path):
                    logger.info(f"Engine compiled by another worker: {os.path.basename(engine_path)}")
                    return

            self._compile_engine(source_path, engine_path)
            if meta_path:
                self._write_metadata(source_path, engine_path, meta_path)

        try:
            os.unlink(lock_path)
        except OSError:
            pass

    def _compile_engine(self, source_path: str, engine_path: str):
        """Compile ONNX/PT → TRT engine. Caller must hold the compile lock."""
        t0 = time.time()
        logger.info(
            f"TRT engine compilation starting: {os.path.basename(source_path)} "
            f"(precision={self.precision}, max_batch={self.max_batch_size})"
        )
        logger.info("Engine compilation may take several minutes — inference unavailable until complete")

        if source_path.endswith(".pt"):
            self._compile_engine_from_pt(source_path, engine_path)
        else:
            self._compile_engine_from_onnx(source_path, engine_path)

        elapsed = time.time() - t0
        logger.info(f"TRT engine compiled in {elapsed:.1f}s → {engine_path}")

    def _compile_engine_from_onnx(self, source_path: str, engine_path: str):
        import onnx
        import tensorrt as trt
        import copy

        original_path = source_path
        onnx_model    = onnx.load(source_path)
        inp_shape     = onnx_model.graph.input[0].type.tensor_type.shape
        dim0          = inp_shape.dim[0]
        is_dynamic    = dim0.HasField("dim_param") or not dim0.HasField("dim_value")

        # Read spatial dims from ONNX; fall back to 640 if dynamic or 0.
        def _static_dim(d):
            return d.dim_value if d.HasField("dim_value") and d.dim_value > 0 else 0
        h = _static_dim(inp_shape.dim[2]) or 640
        w = _static_dim(inp_shape.dim[3]) or 640
        logger.info(f"ONNX input spatial size: {h}×{w}")

        def _prepare_source(lower_fp16: bool) -> str:
            """Return the path to the graph TRT should parse."""
            model = onnx.load(original_path)
            if lower_fp16:
                from onnxconverter_common import float16
                logger.info("Lowering ONNX weights to FP16 for strongly-typed TRT build")
                model = float16.convert_float_to_float16(model, keep_io_types=True)

            if not is_dynamic:
                logger.info("Static batch ONNX — patching to dynamic for TRT optimization profile")
                model = copy.deepcopy(model)
                for t in list(model.graph.input) + list(model.graph.output):
                    # dim_value and dim_param are a protobuf oneof, so assigning
                    # dim_param already clears dim_value. An explicit ClearField
                    # here produces byte-identical output; don't re-add it.
                    t.type.tensor_type.shape.dim[0].dim_param = "batch"
            elif not lower_fp16:
                # Already dynamic and unmodified — parse the file as it stands.
                return original_path

            dyn_path = original_path + ".dyn.onnx"
            onnx.save(model, dyn_path)
            return dyn_path

        logger_trt = trt.Logger(trt.Logger.WARNING)
        builder    = trt.Builder(logger_trt)

        def _parse(path):
            """Parse `path` into a fresh network. Returns (network, errors)."""
            if self.pascal_compat:
                # TRT 8: create_network(0) selects implicit batch, which the ONNX
                # parser rejects outright ("network must have explicit batch").
                net = builder.create_network(
                    1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
                )
            else:
                # TRT 11: implicit batch no longer exists and EXPLICIT_BATCH was
                # removed along with it. Networks are always explicit, always
                # strongly typed.
                net = builder.create_network(0)
            parser = trt.OnnxParser(net, logger_trt)
            with open(path, "rb") as f:
                if parser.parse(f.read()):
                    return net, None
            return None, [str(parser.get_error(i)) for i in range(parser.num_errors)]

        # How precision reaches the engine depends on the TRT generation that
        # this image ships (see PASCAL_COMPAT in the Dockerfiles):
        #
        #   PASCAL_COMPAT=1 (TRT 8)  weak typing. The graph stays FP32 and a
        #       BuilderFlag *permits* the builder to select FP16 kernels. Set
        #       further down, next to the builder config.
        #   PASCAL_COMPAT=0 (TRT 11) strong typing. The per-precision
        #       BuilderFlags were removed, so precision has to come from the
        #       graph itself: lower the weights to FP16 and keep FP32 I/O so
        #       the preprocessing path in _run_trt is unaffected.
        lower_to_fp16 = (not self.pascal_compat) and self.precision == "fp16"

        source_path      = _prepare_source(lower_to_fp16)
        network, errors  = _parse(source_path)

        if network is None and lower_to_fp16:
            # onnxconverter-common keeps some ops (Range, Resize, ...) in FP32,
            # and where one of those feeds a Concat alongside converted
            # siblings, TRT 11 rejects the mixed input types rather than
            # inserting casts the way weak typing used to. It is graph-shaped,
            # not a config error: yolo26n hits it via Range, other YOLO exports
            # do not. Build FP32 rather than failing the compile outright.
            logger.warning(
                "FP16 graph was rejected by the TRT parser — falling back to fp32. "
                "This model mixes FP16 and FP32 at a layer that strongly-typed "
                "TensorRT will not implicitly cast. First error: %s",
                (errors or ["?"])[0][:200],
            )
            # Record what is actually built. Leaving self.precision as fp16
            # would make _check_engine see a permanent fp32 -> fp16 mismatch and
            # recompile on every single load.
            self.precision  = "fp32"
            lower_to_fp16   = False
            source_path     = _prepare_source(False)
            network, errors = _parse(source_path)

        if network is None:
            raise RuntimeError(f"ONNX parse failed: {'; '.join(errors or [])}")

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * 1024 ** 3)
        if self.pascal_compat and self.precision == "fp16":
            # Weak typing: the parsed graph is FP32 and this flag is what allows
            # the builder to pick FP16 kernels. Removed in TRT 11.
            config.set_flag(trt.BuilderFlag.FP16)

        profile = builder.create_optimization_profile()
        inp = network.get_input(0)
        profile.set_shape(inp.name,
                          (1, 3, h, w),
                          (self.max_batch_size, 3, h, w),
                          (self.max_batch_size, 3, h, w))
        config.add_optimization_profile(profile)

        engine_bytes = builder.build_serialized_network(network, config)
        if engine_bytes is None:
            raise RuntimeError("TRT engine build returned None — check TRT logs above")

        tmp_path = engine_path + ".tmp"
        with open(tmp_path, "wb") as f:
            f.write(engine_bytes)
        os.rename(tmp_path, engine_path)   # atomic swap


    def _compile_engine_from_pt(self, source_path: str, engine_path: str):
        import ultralytics
        ultralytics.utils.checks.check_requirements = lambda *a, **kw: True
        from ultralytics import YOLO
        model = YOLO(source_path)
        model.export(format="engine", half=(self.precision == "fp16"),
                     dynamic=True, batch=self.max_batch_size, device=0)
        # Ultralytics writes <source>.engine alongside the .pt
        exported = os.path.splitext(source_path)[0] + ".engine"
        if os.path.exists(exported) and exported != engine_path:
            os.rename(exported, engine_path)

    def _load_trt_direct(self, path: str):
        import tensorrt as trt
        import torch

        runtime = self._get_runtime()
        with open(path, "rb") as f:
            self._trt_engine = runtime.deserialize_cuda_engine(f.read())
        if self._trt_engine is None:
            # Almost always a TRT version mismatch. Without this, the next line
            # raises AttributeError on None and the real cause is invisible.
            raise RuntimeError(
                f"TensorRT could not deserialize {os.path.basename(path)} "
                f"(installed TensorRT: {trt.__version__}). This usually means the "
                f"engine was built by a different TensorRT version. Delete the "
                f".engine and .metadata files to force a rebuild."
            )
        self._trt_ctx = self._trt_engine.create_execution_context()

        n = self._trt_engine.num_io_tensors
        self._trt_in  = None
        self._trt_out = None
        for i in range(n):
            name = self._trt_engine.get_tensor_name(i)
            mode = self._trt_engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self._trt_in = name
            elif mode == trt.TensorIOMode.OUTPUT:
                self._trt_out = name

        shape = self._trt_engine.get_tensor_shape(self._trt_in)
        self._inp_h = shape[2]   # 640
        self._inp_w = shape[3]   # 640

        out_shape = list(self._trt_engine.get_tensor_shape(self._trt_out))
        # Post-NMS engines output (N, K, 6): x1,y1,x2,y2,conf,cls — already filtered.
        # Raw head engines output (N, 4+classes, anchors): external NMS required.
        self._nms_in_model = (len(out_shape) == 3 and out_shape[2] == 6)
        logger.info(f"TRT output {out_shape} → {'post-NMS (skipping external NMS)' if self._nms_in_model else 'raw head (applying external NMS)'}")

        self._ensure_stream()

        # Warm-up
        dummy = torch.zeros(1, 3, self._inp_h, self._inp_w, device="cuda")
        self._trt_forward(dummy)
        logger.info(f"TRT engine ready — input '{self._trt_in}' {list(shape)}")
        self.model = True   # sentinel so callers know model is loaded

    def _load_ultralytics(self, path: str):
        from ultralytics import YOLO
        m = YOLO(path, task="detect")
        if path.endswith(".pt"):
            if self.precision == "fp32":
                m.model.float()
            elif self.precision == "fp16":
                m.model.half()
            elif self.precision == "bf16":
                m.model.bfloat16()
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        m(dummy, device=self.device, half=(self.precision == "fp16"), verbose=False)
        self.model = m
        self._trt_ctx = None

    # ── Inference ──────────────────────────────────────────────────────────

    def run_inference(self, frames_np: np.ndarray) -> np.ndarray:
        if self._trt_ctx is not None:
            return self._run_trt(frames_np)
        return self._run_ultralytics(frames_np)

    # ── Direct TRT path ────────────────────────────────────────────────────

    def _ensure_stream(self):
        """Create the dedicated preprocessing stream.

        Lazy so that callers which set _inp_h/_inp_w without going through
        _load_trt_direct (the unit tests) still work.
        """
        import torch
        if getattr(self, "_stream", None) is None:
            self._stream = torch.cuda.Stream()

    def _trt_forward(self, inp: "torch.Tensor") -> "torch.Tensor":
        """Execute the TRT engine on an (N,3,H,W) CUDA tensor."""
        import torch

        # TensorRT reads the input through a raw data_ptr() and assumes a
        # contiguous NCHW buffer. A permuted tensor (any HWC source) is a view
        # with non-contiguous strides, so handing it over directly makes the
        # engine read the wrong layout and silently return nothing. Batching
        # used to hide this, because torch.cat materialises a contiguous copy;
        # a single-frame batch passed straight through does not.
        if not inp.is_contiguous():
            inp = inp.contiguous()

        batch = inp.shape[0]
        self._trt_ctx.set_input_shape(self._trt_in, list(inp.shape))
        out_shape = list(self._trt_ctx.get_tensor_shape(self._trt_out))
        out_shape[0] = batch
        out = torch.empty(out_shape, dtype=torch.float32, device="cuda")

        self._trt_ctx.set_tensor_address(self._trt_in,  inp.data_ptr())
        self._trt_ctx.set_tensor_address(self._trt_out, out.data_ptr())

        # Run on the dedicated preprocessing stream rather than the default one.
        # Preprocessing was issued there, so the engine is already ordered behind
        # it; wait_stream additionally covers work still pending on the default
        # stream, such as the output allocation above. TensorRT 11 warns
        # explicitly that enqueueV3 on the default stream forces extra
        # cudaStreamSynchronize calls.
        stream = getattr(self, "_stream", None)
        if stream is None:
            self._trt_ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
            torch.cuda.synchronize()
        else:
            stream.wait_stream(torch.cuda.default_stream())
            self._trt_ctx.execute_async_v3(stream.cuda_stream)
            stream.synchronize()
        return out

    def _run_trt(self, frames_np) -> np.ndarray:
        import torch
        import torch.nn.functional as F
        import torchvision.ops as tvops

        if isinstance(frames_np, np.ndarray):
            frames = list(frames_np) if frames_np.ndim == 4 else [frames_np]
        else:
            frames = frames_np
        batch  = len(frames)

        # ── Preprocess ────────────────────────────────────────────────────
        # Which wire format arrives is decided by Frigate's model.input_tensor
        # and model.input_dtype, and they cost very different amounts:
        #
        #   float32 CHW  uploaded to the GPU as-is. For input_dtype: float the
        #                pixels are already normalised, so the CPU never touches
        #                them -- no transpose, no uint8 round trip.
        #   uint8        uploaded directly and scaled on the GPU. Staging this
        #                through a pinned buffer was benchmarked and is slower:
        #                the extra host-side copy costs more than the async DMA
        #                saves.
        #   float32 HWC  uploaded directly and permuted on the GPU.
        #
        # Normalization follows the wire dtype rather than pixel values: a
        # near-black uint8 frame reads as already-normalised to a value sniff,
        # which would skip the divide and hand the model a 255x over-bright
        # image. The float paths keep a value check so Frigate's float_denorm
        # mode, which sends float32 still in 0-255, is scaled correctly.
        self._ensure_stream()
        with torch.cuda.stream(self._stream):
            tensors = []
            for f in frames:
                is_chw = (f.ndim == 3 and f.shape[0] in (1, 3)
                          and f.shape[0] < f.shape[1] and f.shape[0] < f.shape[2])

                if np.issubdtype(f.dtype, np.integer):
                    if is_chw:
                        f = np.ascontiguousarray(np.transpose(f, (1, 2, 0)))
                    t = (torch.as_tensor(np.ascontiguousarray(f), device="cuda")
                              .float().div_(255.0).permute(2, 0, 1).unsqueeze(0))
                else:
                    t = torch.as_tensor(np.ascontiguousarray(f), device="cuda").float()
                    t = t.unsqueeze(0) if is_chw else t.permute(2, 0, 1).unsqueeze(0)
                    if t.max() > 1.5:        # float_denorm: float32 still in 0-255
                        t.div_(255.0)
                tensors.append(t)

            inp = tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=0)
        if inp.shape[2] != self._inp_h or inp.shape[3] != self._inp_w:
            inp = F.interpolate(inp, (self._inp_h, self._inp_w),
                                mode="bilinear", align_corners=False)

        raw = self._trt_forward(inp)

        # ── Post-NMS path: (N, K, 6) — x1,y1,x2,y2,conf,cls in pixel coords
        if self._nms_in_model:
            raw_np = raw.cpu().numpy()
            results = []
            for i in range(batch):
                out  = np.zeros((self.max_dets, 6), np.float32)
                dets = raw_np[i]                       # (K, 6)
                dets = dets[dets[:, 4] > 0]            # drop zero-conf padding rows
                n    = min(len(dets), self.max_dets)
                if n:
                    d = dets[:n]
                    out[:n, 0] = d[:, 5]               # cls
                    out[:n, 1] = d[:, 4]               # conf
                    out[:n, 2] = d[:, 1] / self._inp_h # y1 normalised
                    out[:n, 3] = d[:, 0] / self._inp_w # x1 normalised
                    out[:n, 4] = d[:, 3] / self._inp_h # y2 normalised
                    out[:n, 5] = d[:, 2] / self._inp_w # x2 normalised
                results.append(out)
            return np.array(results)

        # ── Raw head path: decode (cx,cy,w,h) → (x1,y1,x2,y2), apply NMS ──
        scale = float(self._inp_h)
        b  = raw[:, :4, :] / scale                 # (N, 4, 8400)
        cx, cy, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        x1 = (cx - w * 0.5).clamp(0.0, 1.0)
        y1 = (cy - h * 0.5).clamp(0.0, 1.0)
        x2 = (cx + w * 0.5).clamp(0.0, 1.0)
        y2 = (cy + h * 0.5).clamp(0.0, 1.0)
        boxes = torch.stack([x1, y1, x2, y2], dim=2)  # (N, 8400, 4)

        conf, cls_idx = raw[:, 4:, :].max(dim=1)  # (N, 8400) each

        CONF = 0.25
        IOU  = 0.45

        # Zero-area boxes (center clamped to boundary) cause divide-by-zero in
        # Norfair's distance function and crash the camera processor process.
        area = (x2 - x1) * (y2 - y1)              # (N, 8400)

        results = []
        for i in range(batch):
            mask = (conf[i] > CONF) & (area[i] > 0)
            if not mask.any():
                results.append(np.zeros((self.max_dets, 6), np.float32))
                continue

            bi  = boxes[i][mask]      # (K, 4)
            ci  = conf[i][mask]       # (K,)
            cli = cls_idx[i][mask]    # (K,)

            keep = tvops.nms(bi, ci, IOU)[:self.max_dets]
            n    = len(keep)

            # Single CPU transfer for all kept detections
            kept = torch.stack([
                cli[keep].float(),
                ci[keep],
                bi[keep, 1],   # y1 (Frigate: class, conf, y1, x1, y2, x2)
                bi[keep, 0],   # x1
                bi[keep, 3],   # y2
                bi[keep, 2],   # x2
            ], dim=1).cpu().numpy()

            out = np.zeros((self.max_dets, 6), np.float32)
            out[:n] = kept
            results.append(out)

        return np.array(results)

    # ── Ultralytics fallback ───────────────────────────────────────────────

    def _run_ultralytics(self, frames_np) -> np.ndarray:
        if isinstance(frames_np, np.ndarray):
            frames = list(frames_np) if frames_np.ndim == 4 else [frames_np]
        else:
            frames = frames_np
        n      = len(frames)
        zeros  = np.zeros((n, self.max_dets, 6), np.float32)

        if self.model is None or self.model is True:
            return zeros

        try:
            input_data = []
            for f in frames:
                if (f.ndim == 3 and f.shape[0] in (1, 3)
                        and f.shape[0] < f.shape[1] and f.shape[0] < f.shape[2]):
                    f = np.transpose(f, (1, 2, 0))
                    if f.dtype in (np.float32, np.float64):
                        f = (f * 255).clip(0, 255).astype(np.uint8)
                input_data.append(f)

            results = self.model(input_data, device=self.device,
                                 half=(self.precision == "fp16"), verbose=False)

            batch_results = []
            for res in results:
                out   = np.zeros((self.max_dets, 6), np.float32)
                boxes = res.boxes
                if boxes is not None and len(boxes):
                    k     = min(len(boxes), self.max_dets)
                    xyxyn = boxes.xyxyn.cpu().numpy()[:k]
                    conf  = boxes.conf.cpu().numpy()[:k]
                    cls   = boxes.cls.cpu().numpy()[:k]
                    out[:k, 0] = cls
                    out[:k, 1] = conf
                    out[:k, 2] = xyxyn[:, 1]
                    out[:k, 3] = xyxyn[:, 0]
                    out[:k, 4] = xyxyn[:, 3]
                    out[:k, 5] = xyxyn[:, 2]
                    order = out[:k, 1].argsort()[::-1]
                    out[:k] = out[:k][order]
                batch_results.append(out)
            return np.array(batch_results)
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return zeros
