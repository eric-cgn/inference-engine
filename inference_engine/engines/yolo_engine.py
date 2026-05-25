import logging
import os
import numpy as np

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
        Used when no compiled engine exists yet. Slower due to Python-side
        pre/post-processing; intended only for the first run before optimize.py
        has compiled an engine.
    """

    # ── Shared TRT runtime (one per process) ──────────────────────────────
    _trt_runtime = None

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
            engine_path = path + ".engine"

            if os.path.exists(engine_path):
                path = engine_path
            elif not os.path.splitext(path)[1]:
                aliased = path + ".onnx"
                if os.path.exists(path) and not os.path.exists(aliased):
                    os.symlink(os.path.basename(path), aliased)
                path = aliased

            mtime = os.path.getmtime(path) if os.path.exists(path) else 0.0
            if (self.model is not None and self.model_path == path
                    and self.model_mtime == mtime):
                logger.info(f"Model {os.path.basename(original_path)} already loaded.")
                return True

            logger.info(f"Loading YOLO model: {path}  device={self.device}  precision={self.precision}")

            if path.endswith(".engine"):
                self._load_trt_direct(path)
            else:
                self._load_ultralytics(path)

            self.model_name  = os.path.basename(original_path)
            self.model_path  = path
            self.model_mtime = mtime
            logger.info(f"Model {self.model_name} loaded.")
            return True
        except Exception as e:
            logger.error(f"Failed to load model {path}: {e}")
            return False

    def _load_trt_direct(self, path: str):
        import tensorrt as trt
        import torch

        runtime = self._get_runtime()
        with open(path, "rb") as f:
            self._trt_engine = runtime.deserialize_cuda_engine(f.read())
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

    def _trt_forward(self, inp: "torch.Tensor") -> "torch.Tensor":
        """Execute the TRT engine on an (N,3,H,W) CUDA tensor."""
        import torch

        batch = inp.shape[0]
        self._trt_ctx.set_input_shape(self._trt_in, list(inp.shape))
        out_shape = list(self._trt_ctx.get_tensor_shape(self._trt_out))
        out_shape[0] = batch
        out = torch.empty(out_shape, dtype=torch.float32, device="cuda")

        self._trt_ctx.set_tensor_address(self._trt_in,  inp.data_ptr())
        self._trt_ctx.set_tensor_address(self._trt_out, out.data_ptr())
        self._trt_ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        return out

    def _run_trt(self, frames_np: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn.functional as F
        import torchvision.ops as tvops

        frames = list(frames_np) if frames_np.ndim == 4 else [frames_np]
        batch  = len(frames)
        zeros  = np.zeros((batch, self.max_dets, 6), np.float32)

        # ── Preprocess ────────────────────────────────────────────────────
        tensors = []
        for f in frames:
            # Frigate may send CHW float32 (nchw input_tensor mode)
            if (f.ndim == 3 and f.shape[0] in (1, 3)
                    and f.shape[0] < f.shape[1] and f.shape[0] < f.shape[2]):
                f = np.transpose(f, (1, 2, 0))
                if f.dtype in (np.float32, np.float64):
                    f = (f * 255).clip(0, 255).astype(np.uint8)
            t = torch.as_tensor(f.copy(), device="cuda").float().div_(255.0)
            t = t.permute(2, 0, 1).unsqueeze(0)   # (1, C, H, W)
            tensors.append(t)

        inp = torch.cat(tensors, dim=0)            # (N, C, H, W)
        if inp.shape[2] != self._inp_h or inp.shape[3] != self._inp_w:
            inp = F.interpolate(inp, (self._inp_h, self._inp_w),
                                mode="bilinear", align_corners=False)

        # ── TRT forward ───────────────────────────────────────────────────
        raw = self._trt_forward(inp)               # (N, 45, 8400)

        # ── Decode boxes: (cx,cy,w,h) pixel → (x1,y1,x2,y2) normalised ──
        scale = float(self._inp_h)
        b = raw[:, :4, :] / scale                  # (N, 4, 8400)
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

    def _run_ultralytics(self, frames_np: np.ndarray) -> np.ndarray:
        frames = list(frames_np) if frames_np.ndim == 4 else [frames_np]
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
