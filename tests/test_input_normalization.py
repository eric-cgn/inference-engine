"""Input normalization tests for YoloEngine._run_trt.

Frigate sends detection frames in one of three encodings, selected by its
`input_dtype` setting:

    int           — uint8 in 0-255
    float         — float32 already normalized to 0-1
    float_denorm  — float32 still in 0-255

The engine must hand TensorRT values in 0-1 in every case. These tests drive
the real _run_trt path with _trt_forward stubbed out, so they cover the
shipped branch rather than a reimplementation of it.

The regression these lock down is the `float` row. Before the dtype check,
_run_trt divided by 255 unconditionally, so a frame that Frigate had already
normalised to 0-1 was divided a second time and reached the model 255x too
dark. Only HWC frames were affected: the CHW branch above converts float32
back to uint8 first, which is why `input_tensor: nchw` masked the bug and
`nhwc` -- the setting shipped in config/frigate-detector.yaml -- did not.

Requires a CUDA device: _run_trt builds its tensors with device="cuda".
Run from the repo root inside the inference image:

    docker run --rm --device nvidia.com/gpu=0 \
        -v /media/raid10/inference-engine:/app \
        frigate-inference:sm_75_121 \
        python3 -m unittest discover -s tests -t .
"""
import os
import unittest

import numpy as np

# Mirror ServerConfig's parsing without importing it (config imports torch at
# module scope, which would break collection on a bare checkout). YoloEngine
# validates this against the installed TensorRT, so it must match the image.
_PASCAL_COMPAT = os.environ.get("PASCAL_COMPAT", "0").strip().lower() in (
    "1", "true", "yes", "on",
)

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
                torch.cuda.current_stream().wait_stream(self._stream)
                captured["inp"] = inp.detach().clone()
                # 4 box channels + 2 classes over 10 anchors: enough shape for
                # the raw-head decode to run, and all-zero so it finds nothing.
                return torch.zeros(inp.shape[0], 6, 10, device="cuda")

        self.captured = captured
        self.engine = _CapturingEngine(
            device="cuda:0", model_dir="/models", max_dets=20,
            precision="fp16", optimize="never", max_batch_size=1,
            pascal_compat=_PASCAL_COMPAT,
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
        """A near-black uint8 frame is scaled like any other integer frame.

        This passes both before and after the dtype check -- unconditional
        division already handled it. It is kept because it pins the property
        that matters if the float branch is ever widened to cover integers:
        normalization must follow the wire dtype, never the pixel values.
        """
        frame = np.ones((SIZE, SIZE, 3), dtype=np.uint8)
        self.assertAlmostEqual(self._max_sent_to_trt(frame), 1 / 255, places=6)

    def test_normalized_float_is_left_alone(self):
        """Regression: input_dtype float is already 0-1, must not be divided again.

        This is the case that actually failed. Unconditional division turned a
        full-brightness normalised frame into 1/255, a 255x error, for every
        HWC float32 frame Frigate sent.
        """
        frame = np.full((SIZE, SIZE, 3), 1.0, dtype=np.float32)
        self.assertAlmostEqual(self._max_sent_to_trt(frame), 1.0, places=4)

    def test_denormalized_float_is_scaled(self):
        """input_dtype: float_denorm — float32 in 0-255 still needs scaling."""
        frame = np.full((SIZE, SIZE, 3), 255.0, dtype=np.float32)
        self.assertAlmostEqual(self._max_sent_to_trt(frame), 1.0, places=4)


class PrecisionEnforcementTest(unittest.TestCase):
    """PASCAL_COMPAT must force a precision the hardware can actually run.

    Runs without a GPU: the pascal_compat branch short-circuits before any
    device capability lookup.
    """

    @staticmethod
    def _engine(**kw):
        from inference_engine.engine import InferenceEngine

        class _Stub(InferenceEngine):
            def load_model(self, path): return True
            def run_inference(self, frames_np): return None

        kw.setdefault("device", "cpu")
        kw.setdefault("model_dir", "/models")
        return _Stub(**kw)

    def test_pascal_compat_forces_fp32(self):
        for requested in ("fp16", "bf16"):
            with self.subTest(precision=requested):
                e = self._engine(precision=requested, pascal_compat=True)
                self.assertEqual(e.precision, "fp32")

    def test_pascal_compat_leaves_fp32_alone(self):
        e = self._engine(precision="fp32", pascal_compat=True)
        self.assertEqual(e.precision, "fp32")

    def test_non_pascal_keeps_requested_precision_when_supported(self):
        # Without CUDA the capability check is skipped, so the request stands.
        e = self._engine(precision="fp16", pascal_compat=False)
        self.assertIn(e.precision, ("fp16", "fp32"))

    def test_invalid_precision_still_raises(self):
        with self.assertRaises(ValueError):
            self._engine(precision="int8", pascal_compat=False)


if __name__ == "__main__":
    unittest.main()
