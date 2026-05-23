"""
Abstract base class for inference engines.

To add a new backend (ONNX Runtime, OpenVINO, custom TRT pipeline, etc.):
  1. Create engines/<name>_engine.py
  2. Subclass InferenceEngine and implement load_model() + run_inference()
  3. Register it in engines/__init__.py create_engine()
  4. Set engine_type in inference.yaml
"""
from abc import ABC, abstractmethod
import numpy as np

_VALID_PRECISIONS = {"fp32", "fp16", "bf16"}


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
                 max_dets: int = 20, precision: str = "fp32"):
        if precision not in _VALID_PRECISIONS:
            raise ValueError(
                f"precision must be one of {_VALID_PRECISIONS}, got '{precision}'"
            )
        self.device     = device
        self.model_dir  = model_dir
        self.max_dets   = max_dets
        self.precision  = precision
        self.model      = None
        self.model_name = None
        self.model_path = None
        self.model_mtime = None

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
