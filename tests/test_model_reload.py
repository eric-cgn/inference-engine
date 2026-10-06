"""Repeated model_request for the loaded model must not re-hash it.

Frigate sends model_request again after every ZMQ timeout. Hashing a large
ONNX on each one blocked the batch thread long enough to cause more timeouts.
The check is skipped while the source, engine and metadata files are unchanged.

Run inside the image, from the repo root:

    docker run --rm --device nvidia.com/gpu=0 -v $PWD:/app -w /app \
        frigate-inference:sm_75_121 python3 -m unittest -v tests.test_model_reload
"""
import os
import shutil
import tempfile
import unittest

# Must match the image: YoloEngine refuses to start when PASCAL_COMPAT
# disagrees with the installed TensorRT, so hardcoding it fails on sm_61.
_PASCAL_COMPAT = os.environ.get("PASCAL_COMPAT", "0").strip().lower() in ("1", "true", "yes", "on")

try:
    from inference_engine.engines.yolo_engine import YoloEngine
    _IMPORT_ERROR = None
except Exception as e:      # tensorrt/torch absent (lint-only checkout)
    YoloEngine = None
    _IMPORT_ERROR = e


@unittest.skipIf(YoloEngine is None, f"engine not importable: {_IMPORT_ERROR}")
class ModelReloadTest(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        for name in ("m.onnx", "m.engine", "m.metadata"):
            with open(os.path.join(self.dir, name), "wb") as f:
                f.write(b"x" * 16)

        checks, loads = [], []

        class _Stub(YoloEngine):
            def _check_engine(self, source, engine_path, meta_path):
                checks.append(engine_path)
                return "use", None

            def _load_trt_direct(self, path):
                loads.append(path)
                self.model = True

        self.checks, self.loads = checks, loads
        self.engine = _Stub(device="cpu", model_dir=self.dir, max_dets=20,
                            precision="fp32", optimize="always", max_batch_size=1,
                            pascal_compat=_PASCAL_COMPAT)
        self.path = os.path.join(self.dir, "m.onnx")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_repeat_request_skips_check(self):
        for _ in range(5):
            self.assertTrue(self.engine.load_model(self.path))
        self.assertEqual(len(self.checks), 1)
        self.assertEqual(len(self.loads), 1)

    def test_changed_source_is_checked_again(self):
        self.engine.load_model(self.path)
        with open(self.path, "ab") as f:
            f.write(b"y")
        self.engine.load_model(self.path)
        self.assertEqual(len(self.checks), 2)

    def test_changed_engine_is_checked_again(self):
        self.engine.load_model(self.path)
        st = os.stat(os.path.join(self.dir, "m.engine"))
        os.utime(os.path.join(self.dir, "m.engine"), ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
        self.engine.load_model(self.path)
        self.assertEqual(len(self.checks), 2)

    def test_failed_load_is_not_remembered(self):
        def boom(path):
            raise RuntimeError("deserialize failed")
        self.engine._load_trt_direct = boom
        self.assertFalse(self.engine.load_model(self.path))
        self.assertFalse(self.engine.load_model(self.path))
        self.assertEqual(len(self.checks), 2)


if __name__ == "__main__":
    unittest.main()
