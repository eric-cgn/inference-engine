import os
import logging
import torch
import yaml

logger = logging.getLogger("inference_config")

_DEFAULTS = {
    "endpoint":       "ipc:///run/zmq/detector.sock",
    "model_dir":      "/models",
    "device":         None,   # resolved below
    "precision":      "fp32",
    "num_workers":    1,
    "max_batch_size": 16,
}


class ServerConfig:
    def __init__(self):
        # Sensible defaults — all overridable via inference.yaml or env vars.
        self.endpoint       = os.environ.get("ZMQ_ENDPOINT", _DEFAULTS["endpoint"])
        self.model_dir      = os.environ.get("MODEL_DIR",    _DEFAULTS["model_dir"])
        self.device         = os.environ.get("DEVICE",       "cuda:0" if torch.cuda.is_available() else "cpu")
        self.precision      = os.environ.get("PRECISION",    _DEFAULTS["precision"])
        self.engine_type    = os.environ.get("ENGINE_TYPE",  "yolo")
        self.num_workers    = int(os.environ.get("NUM_WORKERS",    _DEFAULTS["num_workers"]))
        self.max_batch_size = int(os.environ.get("MAX_BATCH_SIZE", _DEFAULTS["max_batch_size"]))
        self.max_dets       = 20

        # backend_endpoint is an internal ZMQ detail, not user-facing.
        # Always inproc:// — workers and broker share the same container and context.
        self.backend_endpoint = "inproc://workers"

        config_path = os.environ.get("CONFIG_PATH", "/config/inference.yaml")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    data = yaml.safe_load(f) or {}
                # YAML file overrides defaults; env vars were already set above
                # and take precedence over the file if explicitly provided.
                if "ZMQ_ENDPOINT"   not in os.environ: self.endpoint       = data.get("endpoint",       self.endpoint)
                if "MODEL_DIR"      not in os.environ: self.model_dir      = data.get("model_dir",      self.model_dir)
                if "DEVICE"         not in os.environ: self.device         = data.get("device",         self.device)
                if "PRECISION"      not in os.environ: self.precision      = data.get("precision",      self.precision)
                if "ENGINE_TYPE"    not in os.environ: self.engine_type    = data.get("engine_type",    self.engine_type)
                if "NUM_WORKERS"    not in os.environ: self.num_workers    = int(data.get("num_workers",    self.num_workers))
                if "MAX_BATCH_SIZE" not in os.environ: self.max_batch_size = int(data.get("max_batch_size", self.max_batch_size))
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.error(f"Failed to load config file {config_path}: {e}")
        else:
            logger.info(f"No config file at {config_path} — using defaults / env vars.")

    def summary(self):
        return {
            "endpoint":    self.endpoint,
            "device":      self.device,
            "precision":   self.precision,
            "engine_type": self.engine_type,
            "num_workers": self.num_workers,
            "max_batch":   self.max_batch_size,
        }
