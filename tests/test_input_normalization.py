"""Input normalization tests for YoloEngine._run_trt.

Frigate sends detection frames in one of three encodings, selected by its
`input_dtype` setting:

    int           — uint8 in 0-255
    float         — float32 already normalized to 0-1
    float_denorm  — float32 still in 0-255

The engine must hand TensorRT values in 0-1 in every case. These tests drive
the real _run_trt path with _trt_forward stubbed out, so they cover the
shipped branch rather than a reimplementation of it.

Requires a CUDA device: _run_trt builds its tensors with device="cuda".
Run from the repo root inside the inference image:

    docker run --rm --device nvidia.com/gpu=0 \
        -v /media/raid10/inference-engine:/app \
        frigate-inference:sm_75plus \
        python3 -m unittest discover -s tests -t .
"""
import unittest

import numpy as np

try:
    import torch
    _CUDA = torch.cuda.is_available()
except ImportError:          # torch absent entirely (e.g. lint-only checkout)
    torch = None
    _CUDA = False

if _CUDA:
    from inference_engine.engines.yolo_engine import YoloEngine

# Frame side length. Kept small for speed, but must not be 1 or 3 — those
# would trip the CHW-detection branch at the top of _run_trt.
SIZE = 32


@unittest.skipUnless(_CUDA, "requires a CUDA device (_run_trt is cuda-only)")
class InputNormalizationTest(unittest.TestCase):
    """TRT must receive 0-1 values whatever dtype arrived over ZMQ."""

    def setUp(self):
        captured = {}

        class _CapturingEngine(YoloEngine):
            def _trt_forward(self, inp):
                captured["inp"] = inp.detach().clone()
                # 4 box channels + 2 classes over 10 anchors: enough shape for
                # the raw-head decode to run, and all-zero so it finds nothing.
                return torch.zeros(inp.shape[0], 6, 10, device="cuda")

        self.captured = captured
        self.engine = _CapturingEngine(
            device="cuda:0", model_dir="/models", max_dets=20,
            precision="fp16", optimize="never", max_batch_size=1,
        )
        # Select the direct-TRT path without deserializing a real engine.
        self.engine._trt_ctx = object()
        self.engine._nms_in_model = False
        self.engine._inp_h = self.engine._inp_w = SIZE

    def _max_sent_to_trt(self, frame):
        """Return the largest value actually handed to the TRT engine."""
        self.captured.clear()
        self.engine._run_trt([frame])
        return float(self.captured["inp"].max())

    def test_uint8_is_scaled(self):
        """input_dtype: int — full-brightness uint8 lands at 1.0."""
        frame = np.full((SIZE, SIZE, 3), 255, dtype=np.uint8)
        self.assertAlmostEqual(self._max_sent_to_trt(frame), 1.0, places=4)

    def test_near_black_uint8_is_still_scaled(self):
        """Regression: normalization must not be decided from pixel values.

        A frame whose brightest pixel is 1/255 looks pre-normalized to a
        `t.max() > 1.5` check, so the divide gets skipped and the model
        receives a 255x over-bright image. The wire dtype is already in the
        ZMQ header, so the decision never needs to inspect the data.
        """
        frame = np.ones((SIZE, SIZE, 3), dtype=np.uint8)
        self.assertAlmostEqual(self._max_sent_to_trt(frame), 1 / 255, places=6)

    def test_normalized_float_is_left_alone(self):
        """input_dtype: float — already 0-1, must not be divided again."""
        frame = np.full((SIZE, SIZE, 3), 1.0, dtype=np.float32)
        self.assertAlmostEqual(self._max_sent_to_trt(frame), 1.0, places=4)

    def test_denormalized_float_is_scaled(self):
        """input_dtype: float_denorm — float32 in 0-255 still needs scaling."""
        frame = np.full((SIZE, SIZE, 3), 255.0, dtype=np.float32)
        self.assertAlmostEqual(self._max_sent_to_trt(frame), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
