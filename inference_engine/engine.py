"""
Abstract base class for inference engines.

To add a new backend (ONNX Runtime, OpenVINO, custom TRT pipeline, etc.):
  1. Create engines/<name>_engine.py
  2. Subclass InferenceEngine and implement load_model() + run_inference()
  3. Register it in engines/__init__.py create_engine()
  4. Set engine_type in inference.yaml
"""
from abc import ABC, abstractmethod
import logging
import numpy as np

logger = logging.getLogger("inference_engine")

_VALID_PRECISIONS = {"fp32", "fp16", "bf16"}

# Minimum CUDA compute capability for each precision. FP16 has Tensor Core
# support from Turing (7.5); BF16 hardware does not exist before Ampere (8.0).
# Pascal can execute FP16 but at 1/64 rate, so it is treated as unsupported.
_MIN_CAPABILITY = {"fp16": (7, 5), "bf16": (8, 0)}


class InferenceEngine(ABC):
    """
    Abstract inference engine. Handles the ZMQ pipeline contract:
    load_model() is called once (on model_request), run_inference() is
    called per batch. All engines must accept the same constructor args
    so the factory can instantiate any of them uniformly.

    precision semantics:
        fp32 - Full precision. Required for Pascal (sm_6.1) and older.
        fp16 - Half precision. Tensor Cores on Turing (sm_75)+. ~2x throughput.
        bf16 - BFloat16. Ampere (sm_80)+ only.
    """

    def __init__(self, device: str, model_dir: str,
                 max_dets: int = 20, precision: str = "fp32",
                 optimize: str = "if_present", max_batch_size: int = 1,
                 pascal_compat: bool = False):
        if precision not in _VALID_PRECISIONS:
            raise ValueError(
                f"precision must be one of {_VALID_PRECISIONS}, got '{precision}'"
            )
        self.pascal_compat   = pascal_compat
        precision            = self._enforce_precision(precision)
        self.device          = device
        self.model_dir       = model_dir
        self.max_dets        = max_dets
        self.precision       = precision
        self.optimize        = optimize
        self.max_batch_size  = max_batch_size
        self.model           = None
        self.model_name      = None
        self.model_path      = None
        self.model_mtime     = None
        self._compiling      = False
        self._compile_failed = False

    def _enforce_precision(self, precision: str) -> str:
        """
        Downgrade `precision` to what the hardware can actually execute.

        Warns rather than raises: an unsupported precision is a misconfiguration
        that should not take an NVR offline. The engine still runs, just at a
        precision the GPU supports.
        """
        if self.pascal_compat and precision != "fp32":
            logger.warning(
                f"PASCAL_COMPAT=1 but precision='{precision}' — Pascal has no Tensor "
                f"Cores and executes FP16 at 1/64 rate. Using fp32 instead."
            )
            return "fp32"

        need = _MIN_CAPABILITY.get(precision)
        if need is None:
            return precision

        cap = self._device_capability()
        if cap is None:
            # No CUDA device visible (CPU mode, or a unit test on a bare host).
            # Nothing to validate against; leave the request untouched.
            return precision

        if cap < need:
            fallback = "fp16" if precision == "bf16" and cap >= _MIN_CAPABILITY["fp16"] else "fp32"
            logger.warning(
                f"precision='{precision}' requires compute capability "
                f"{need[0]}.{need[1]}+, but this GPU is {cap[0]}.{cap[1]}. "
                f"Using {fallback} instead."
            )
            return fallback
        return precision

    @staticmethod
    def _device_capability():
        """(major, minor) of the active CUDA device, or None if unavailable."""
        try:
            import torch
            if not torch.cuda.is_available():
                return None
            return torch.cuda.get_device_capability()
        except Exception:
            return None

    @abstractmethod
    def load_model(self, path: str) -> bool:
        """
        Load a model from `path`. Returns True on success.
        Must set self.model and self.model_name on success.
        """

    @abstractmethod
    def run_inference(self, frames_np: np.ndarray) -> np.ndarray:
        """
        Run inference on a batch of frames.

        Args:
            frames_np: (N, H, W, 3) uint8 numpy array.

        Returns:
            (N, max_dets, 6) float32 array in Frigate format:
            [class_id, confidence, y_min, x_min, y_max, x_max]
        """
