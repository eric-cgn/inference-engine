"""
Engine factory. Add new backends here.
"""
from ..engine import InferenceEngine


def create_engine(engine_type: str, **kwargs) -> InferenceEngine:
    """
    Instantiate an engine by type name.

    Args:
        engine_type: One of "yolo" (more to come).
        **kwargs:    Passed directly to the engine constructor:
                     device, model_dir, max_dets, precision.
    """
    if engine_type == "yolo":
        from .yolo_engine import YoloEngine
        return YoloEngine(**kwargs)

    raise ValueError(
        f"Unknown engine_type '{engine_type}'. "
        f"Valid options: 'yolo'"
    )
