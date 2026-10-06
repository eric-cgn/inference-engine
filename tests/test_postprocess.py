"""Raw-head postprocessing in YoloEngine._run_trt.

The decode, confidence filter and NMS run over the whole batch at once, with
each frame's boxes shifted apart so NMS never compares boxes from different
frames. These tests pin that behaviour with a stubbed _trt_forward, so no
TensorRT engine is needed -- only a CUDA device.

Run inside the image, from the repo root:

    docker run --rm --device nvidia.com/gpu=0 -v $PWD:/app \
        frigate-inference:sm_75_121 -m unittest -v tests.test_postprocess
"""
import os
import unittest

import numpy as np

_PASCAL_COMPAT = os.environ.get("PASCAL_COMPAT", "0").strip().lower() in ("1", "true", "yes", "on")

try:
    import torch
    _CUDA = torch.cuda.is_available()
except ImportError:
    torch = None
    _CUDA = False

if _CUDA:
    from inference_engine.engines.yolo_engine import YoloEngine

SIZE = 32       # input side; anchors are given in these pixel units
CLASSES = 3
MAX_DETS = 20


def _anchor(cx, cy, w, h, cls, conf):
    """One raw-head anchor column: cx, cy, w, h, then one score per class."""
    col = [cx, cy, w, h] + [0.0] * CLASSES
    col[4 + cls] = conf
    return col


@unittest.skipUnless(_CUDA, "requires CUDA device (_run_trt is cuda-only)")
class RawHeadPostprocessTest(unittest.TestCase):

    def _run(self, per_frame_anchors):
        """Run _run_trt on len(per_frame_anchors) frames whose raw output is given."""
        n_anchors = max(len(a) for a in per_frame_anchors)
        raw = torch.zeros(len(per_frame_anchors), 4 + CLASSES, n_anchors, device="cuda")
        for i, anchors in enumerate(per_frame_anchors):
            for j, col in enumerate(anchors):
                raw[i, :, j] = torch.tensor(col)

        class _Stub(YoloEngine):
            def _trt_forward(self, inp):
                return raw

        e = _Stub(device="cuda:0", model_dir="/models", max_dets=MAX_DETS,
                  precision="fp16", optimize="never", max_batch_size=4,
                  pascal_compat=_PASCAL_COMPAT)
        e._trt_ctx = object()
        e._nms_in_model = False
        e._inp_h = e._inp_w = SIZE
        frames = [np.zeros((SIZE, SIZE, 3), np.uint8)] * len(per_frame_anchors)
        return e._run_trt(frames)

    def test_identical_boxes_in_different_frames_both_survive(self):
        box = _anchor(16, 16, 8, 8, cls=1, conf=0.9)
        out = self._run([[box], [box]])
        self.assertEqual(out.shape, (2, MAX_DETS, 6))
        for i in range(2):
            self.assertAlmostEqual(float(out[i, 0, 1]), 0.9, places=5)
            self.assertEqual(int(out[i, 0, 0]), 1)
            self.assertEqual(float(out[i, 1, 1]), 0.0)

    def test_overlap_within_a_frame_is_suppressed(self):
        out = self._run([[_anchor(16, 16, 8, 8, cls=0, conf=0.9),
                          _anchor(16.5, 16, 8, 8, cls=2, conf=0.8)]])
        self.assertAlmostEqual(float(out[0, 0, 1]), 0.9, places=5)
        self.assertEqual(float(out[0, 1, 1]), 0.0)

    def test_output_layout_and_order(self):
        """Frigate's (class, conf, y1, x1, y2, x2), normalised, highest conf first."""
        out = self._run([[_anchor(8, 8, 4, 4, cls=0, conf=0.5),
                          _anchor(24, 16, 8, 4, cls=2, conf=0.7)]])
        np.testing.assert_allclose(out[0, 0], [2, 0.7, 14 / 32, 20 / 32, 18 / 32, 28 / 32], atol=1e-5)
        np.testing.assert_allclose(out[0, 1], [0, 0.5, 6 / 32, 6 / 32, 10 / 32, 10 / 32], atol=1e-5)

    def test_low_confidence_and_zero_area_are_dropped(self):
        out = self._run([[_anchor(16, 16, 8, 8, cls=0, conf=0.2),     # below 0.25
                          _anchor(0, 16, 0, 8, cls=0, conf=0.9)],     # zero width
                         [_anchor(16, 16, 8, 8, cls=0, conf=0.9)]])
        self.assertFalse(out[0].any())
        self.assertAlmostEqual(float(out[1, 0, 1]), 0.9, places=5)

    def test_capped_at_max_dets_per_frame(self):
        # 5 x 6 grid of separate boxes = 30 > MAX_DETS, in frame 1 only.
        grid = [_anchor(3 + 5 * c, 3 + 5 * r, 2, 2, cls=0, conf=0.3 + 0.01 * (r * 6 + c))
                for r in range(5) for c in range(6)]
        out = self._run([[_anchor(16, 16, 8, 8, cls=0, conf=0.9)], grid])
        self.assertEqual(int((out[1, :, 1] > 0).sum()), MAX_DETS)
        self.assertTrue(np.all(np.diff(out[1, :, 1]) <= 0))
        self.assertEqual(int((out[0, :, 1] > 0).sum()), 1)


if __name__ == "__main__":
    unittest.main()
