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
                # Preprocessing runs on the engine stream; the real
                # _trt_forward stays on it, but this clone runs on the default
                # stream, so wait or it can read the tensor before it is written.
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


@unittest.skipUnless(_CUDA, "requires a CUDA device (_run_trt is cuda-only)")
class DeclaredFormatTest(unittest.TestCase):
    """A client may declare what the model wants; the engine honours it.

    Without a declaration the engine assumes 0-1, which is what every client
    relied on before these fields existed and what InputNormalizationTest
    above pins down. With one, the model's own convention wins -- a
    float_denorm or int model is no longer silently handed 0-1.
    """

    def setUp(self):
        captured = {}

        class _CapturingEngine(YoloEngine):
            def _trt_forward(self, inp):
                torch.cuda.current_stream().wait_stream(self._stream)
                captured["inp"] = inp.detach().clone()
                return torch.zeros(inp.shape[0], 6, 10, device="cuda")

        self.captured = captured
        self.engine = _CapturingEngine(
            device="cuda:0", model_dir="/models", max_dets=20,
            precision="fp16", optimize="never", max_batch_size=1,
            pascal_compat=_PASCAL_COMPAT,
        )
        self.engine._trt_ctx = object()
        self.engine._nms_in_model = False
        self.engine._inp_h = self.engine._inp_w = SIZE

    def _max_sent(self, frame, declared):
        self.engine.declared_input_dtype = declared
        self.captured.clear()
        self.engine._run_trt([frame])
        return float(self.captured["inp"].max())

    def test_declared_float_scales_uint8(self):
        """input_dtype float: the model wants 0-1, so 255 becomes 1.0."""
        frame = np.full((SIZE, SIZE, 3), 255, dtype=np.uint8)
        self.assertAlmostEqual(self._max_sent(frame, "float"), 1.0, places=4)

    def test_declared_float_denorm_leaves_uint8_unscaled(self):
        """input_dtype float_denorm: the model wants 0-255, so do not divide.

        The engine used to divide here regardless, handing such a model a
        frame 255x too dark.
        """
        frame = np.full((SIZE, SIZE, 3), 255, dtype=np.uint8)
        self.assertAlmostEqual(self._max_sent(frame, "float_denorm"), 255.0, places=2)

    def test_declared_int_leaves_uint8_unscaled(self):
        """input_dtype int: the model takes raw bytes."""
        frame = np.full((SIZE, SIZE, 3), 255, dtype=np.uint8)
        self.assertAlmostEqual(self._max_sent(frame, "int"), 255.0, places=2)

    def test_declaration_is_reversible(self):
        """Clearing the declaration restores the undeclared assumption."""
        frame = np.full((SIZE, SIZE, 3), 255, dtype=np.uint8)
        self.assertAlmostEqual(self._max_sent(frame, "float_denorm"), 255.0, places=2)
        self.assertAlmostEqual(self._max_sent(frame, None), 1.0, places=4)

    def test_chw_uint8_is_permuted_on_the_gpu(self):
        """A CHW frame reaches TRT as NCHW without a host-side transpose."""
        frame = np.zeros((3, SIZE, SIZE), dtype=np.uint8)
        frame[1] = 255                      # green channel only
        self.engine.declared_input_dtype = None
        self.captured.clear()
        self.engine._run_trt([frame])
        inp = self.captured["inp"]
        self.assertEqual(tuple(inp.shape), (1, 3, SIZE, SIZE))
        self.assertAlmostEqual(float(inp[0, 0].max()), 0.0, places=5)
        self.assertAlmostEqual(float(inp[0, 1].max()), 1.0, places=4)

    def test_plan_is_resolved_once_per_wire_format(self):
        """The transform is cached, not re-decided per frame."""
        frame = np.full((SIZE, SIZE, 3), 255, dtype=np.uint8)
        self.engine.declared_input_dtype = None
        self.engine._plans.clear()
        for _ in range(3):
            self.engine._run_trt([frame])
        self.assertEqual(len(self.engine._plans), 1)

        # A different wire format gets its own plan rather than reusing one.
        self.engine._run_trt([np.full((SIZE, SIZE, 3), 1.0, dtype=np.float32)])
        self.assertEqual(len(self.engine._plans), 2)


class BindingIntrospectionTest(unittest.TestCase):
    """Input geometry is read from the binding, not assumed to be NCHW.

    The old code took shape[2] and shape[3] as height and width, so an NHWC
    engine silently resolved to 3x3. Runs without a GPU.
    """

    @staticmethod
    def _introspect(dims):
        from inference_engine.engines.yolo_engine import YoloEngine

        eng = YoloEngine.__new__(YoloEngine)
        eng._trt_in = "images"
        eng._trt_engine = None          # forces the dtype fallback
        eng.declared_input_tensor = None
        return YoloEngine._introspect_binding(eng, dims)

    def test_nchw_binding(self):
        fmt = self._introspect([1, 3, 640, 480])
        self.assertEqual((fmt.layout, fmt.height, fmt.width), ("nchw", 640, 480))

    def test_nhwc_binding(self):
        fmt = self._introspect([1, 640, 480, 3])
        self.assertEqual((fmt.layout, fmt.height, fmt.width), ("nhwc", 640, 480))

    def test_dynamic_batch_is_fine(self):
        fmt = self._introspect([-1, 3, 640, 640])
        self.assertEqual((fmt.layout, fmt.height, fmt.width), ("nchw", 640, 640))

    def test_non_4d_binding_raises(self):
        with self.assertRaises(RuntimeError):
            self._introspect([1, 3, 640])


@unittest.skipUnless(_CUDA, "requires a CUDA device (_run_trt is cuda-only)")
class DeclarationCrossCheckTest(unittest.TestCase):
    """A declared layout that disagrees with the binding must be reported.

    The check runs at plan resolution, not at model load: the model is loaded
    before any request arrives, so at load time nothing has been declared and
    the warning would never fire.
    """

    def setUp(self):
        class _Engine(YoloEngine):
            def _trt_forward(self, inp):
                torch.cuda.current_stream().wait_stream(self._stream)
                return torch.zeros(inp.shape[0], 6, 10, device="cuda")

        self.engine = _Engine(
            device="cuda:0", model_dir="/models", max_dets=20,
            precision="fp16", optimize="never", max_batch_size=1,
            pascal_compat=_PASCAL_COMPAT,
        )
        self.engine._trt_ctx = object()
        self.engine._nms_in_model = False
        self.engine._inp_h = self.engine._inp_w = SIZE
        self.frame = np.full((SIZE, SIZE, 3), 255, dtype=np.uint8)

    def test_mismatched_layout_warns(self):
        self.engine.declared_input_tensor = "nhwc"      # binding is nchw
        with self.assertLogs("yolo_engine", level="WARNING") as cm:
            self.engine._run_trt([self.frame])
        self.assertTrue(any("compiled binding is nchw" in m for m in cm.output))

    def test_matching_layout_is_quiet(self):
        self.engine.declared_input_tensor = "nchw"
        with self.assertLogs("yolo_engine", level="INFO") as cm:
            self.engine._run_trt([self.frame])
        self.assertFalse(any("Client declares" in m for m in cm.output))
