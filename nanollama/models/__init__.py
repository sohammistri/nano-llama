"""
Model registry for nanochat architectures.

Each architecture registers itself via register_model(arch, config_cls, model_cls).
Use build_model_from_config(arch, config_kwargs) to construct models from checkpoint metadata.
"""

MODEL_REGISTRY = {}


def register_model(arch, config_cls, model_cls):
    """Register a model architecture."""
    MODEL_REGISTRY[arch] = (config_cls, model_cls)


def build_model_from_config(arch, config_kwargs):
    """Look up architecture in registry and return (config, model_cls)."""
    if arch not in MODEL_REGISTRY:
        raise ValueError(f"Unknown architecture: {arch!r}. Available: {list(MODEL_REGISTRY.keys())}")
    config_cls, model_cls = MODEL_REGISTRY[arch]
    config = config_cls(**config_kwargs)
    return config, model_cls


# Import model modules to trigger registration
from nanochat.models.gpt import GPT, GPTConfig  # noqa: F401, E402
