import logging
import os
import numpy as np
from ultralytics import YOLO

from ..engine import InferenceEngine

logger = logging.getLogger("yolo_engine")


class YoloEngine(InferenceEngine):
    """
    Ultralytics YOLO inference engine.

    Supports .pt (PyTorch), .onnx, and .engine (TensorRT) model files.
    TRT engine files have precision baked in at export time — the precision
    setting drives the PyTorch model cast and warm-up/inference half= flag
    for .pt/.onnx models, and is ignored for .engine files.

    Run tools/optimize.py to compile a TRT engine for your GPU.
    """

    def load_model(self, path: str) -> bool:
        try:
            original_path = path

            # Prefer a compiled TensorRT engine if one exists alongside the source file.
            engine_path = path + ".engine"
            if os.path.exists(engine_path):
                path = engine_path
            # Extension-free paths are Frigate Plus ONNX pushes — alias them so
            # ultralytics can identify the format.
            elif not os.path.splitext(path)[1]:
                aliased_path = path + ".onnx"
                if os.path.exists(path) and not os.path.exists(aliased_path):
                    os.symlink(os.path.basename(path), aliased_path)
                path = aliased_path

            # Skip reload if the file hasn't changed since last load.
            mtime = os.path.getmtime(path) if os.path.exists(path) else 0.0
            if self.model is not None and self.model_path == path and self.model_mtime == mtime:
                logger.info(f"Model {os.path.basename(original_path)} already loaded and up to date.")
                return True

            logger.info(
                f"Loading YOLO model: {path}  device={self.device}  precision={self.precision}"
            )
            m = YOLO(path, task="detect")

            # Precision cast for PyTorch models only — ONNX/TRT have it baked in.
            if path.endswith(".pt"):
                if self.precision == "fp32":
                    m.model.float()       # required for Pascal (sm_61)
                elif self.precision == "fp16":
                    m.model.half()        # Tensor Cores on Turing+
                elif self.precision == "bf16":
                    m.model.bfloat16()    # Ampere+ only

            # Warm-up: trigger CUDA kernel compilation and VRAM allocation
            # before real frames arrive so the first real batch isn't slow.
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            m(dummy, device=self.device, half=(self.precision == "fp16"), verbose=False)

            self.model       = m
            self.model_name  = os.path.basename(original_path)
            self.model_path  = path
            self.model_mtime = mtime
            logger.info(f"Model {self.model_name} loaded and warmed up.")
            return True
        except Exception as e:
            logger.error(f"Failed to load model {path}: {e}")
            return False

    def run_inference(self, frames_np: np.ndarray) -> np.ndarray:
        n_frames = len(frames_np) if frames_np.ndim == 4 else 1
        zeros    = np.zeros((n_frames, self.max_dets, 6), np.float32)

        if self.model is None:
            return zeros

        try:
            # Ultralytics expects HWC uint8 numpy arrays.
            # Frigate with input_tensor=nchw sends CHW float32 (transposed and
            # normalised). Detect and convert back before LetterBox runs —
            # otherwise LetterBox misinterprets C=3 as H=3 and pads to (640,640,640).
            frames_list = list(frames_np) if frames_np.ndim == 4 else [frames_np]
            input_data = []
            for f in frames_list:
                if (f.ndim == 3
                        and f.shape[0] in (1, 3)
                        and f.shape[0] < f.shape[1]
                        and f.shape[0] < f.shape[2]):
                    f = np.transpose(f, (1, 2, 0))          # CHW → HWC
                    if f.dtype in (np.float32, np.float64):
                        f = (f * 255).clip(0, 255).astype(np.uint8)
                input_data.append(f)

            results = self.model(
                input_data,
                device=self.device,
                half=(self.precision == "fp16"),
                verbose=False,
            )

            batch_results = []
            for res in results:
                out   = np.zeros((self.max_dets, 6), np.float32)
                boxes = res.boxes
                if boxes is not None and len(boxes):
                    n     = min(len(boxes), self.max_dets)
                    xyxyn = boxes.xyxyn.cpu().numpy()[:n]
                    conf  = boxes.conf.cpu().numpy()[:n]
                    cls   = boxes.cls.cpu().numpy()[:n]

                    out[:n, 0] = cls
                    out[:n, 1] = conf
                    out[:n, 2] = xyxyn[:, 1]  # y1
                    out[:n, 3] = xyxyn[:, 0]  # x1
                    out[:n, 4] = xyxyn[:, 3]  # y2
                    out[:n, 5] = xyxyn[:, 2]  # x2

                    order = out[:n, 1].argsort()[::-1]
                    out[:n] = out[:n][order]
                batch_results.append(out)

            return np.array(batch_results)
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return zeros
