"""Qwen3-ASR MLX runtime and JSONL bridge."""

__all__ = [
    "BRIDGE_PROTOCOL_VERSION",
    "DEFAULT_CACHE",
    "DEFAULT_MODEL",
    "SAMPLE_RATE",
    "ProbeContext",
    "Qwen3ASRMLXRuntime",
    "bridge_capabilities",
    "run_runtime_bridge",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import runtime

    value = getattr(runtime, name)
    globals()[name] = value
    return value
