from pathlib import Path

from qwen3_asr_mlx_runtime import DEFAULT_CACHE, DEFAULT_MODEL, ProbeContext, bridge_capabilities


def test_capabilities_do_not_load_model():
    ctx = ProbeContext(
        model=DEFAULT_MODEL,
        cache_dir=Path(DEFAULT_CACHE),
        audio_path=None,
        language=None,
        trust_remote_code=True,
        local_files_only=True,
        max_new_tokens=None,
    )
    payload = bridge_capabilities(ctx, use_cache=True)
    assert payload["type"] == "capabilities"
    assert payload["protocol"] == "qwen3-asr-mlx-jsonl-v1"
    assert payload["model"] == DEFAULT_MODEL
    assert payload["sample_rate"] == 16000
    assert payload["decoder_modes"] == ["sequential", "batched"]
