#!/usr/bin/env python3
"""Reusable Qwen3-ASR MLX runtime.

This module contains the app-independent runtime, model-loading, audio tower,
MRoPE decoder, cached generation, batching, and newline-JSON bridge pieces.
Probe-only CLI stages remain in probe_qwen3_asr_mlx.py.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import dataclasses
import glob
import inspect
import json
import math
import os
import resource
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_MODEL = "Qwen/Qwen3-ASR-0.6B"
DEFAULT_CACHE = Path.home() / ".audio-models"
SAMPLE_RATE = 16_000
AUDIO_PLACEHOLDER = "<|audio_start|><|audio_pad|><|audio_end|>"
ASR_TEXT_TAG = "<asr_text>"
BRIDGE_PROTOCOL_VERSION = "qwen3-asr-mlx-jsonl-v1"
LOG_STREAM = sys.stdout


def set_log_stream(stream: Any) -> None:
    global LOG_STREAM
    LOG_STREAM = stream


class ProbeError(RuntimeError):
    pass


@dataclasses.dataclass
class ProbeContext:
    model: str
    cache_dir: Path
    audio_path: str | None
    language: str | None
    trust_remote_code: bool
    local_files_only: bool
    max_new_tokens: int | None


@dataclasses.dataclass(frozen=True)
class PromptTokenTemplate:
    prefix: np.ndarray
    suffix: np.ndarray
    audio_pad_id: int


def log(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, ensure_ascii=False), file=LOG_STREAM, flush=True)


def bridge_capabilities(ctx: ProbeContext, use_cache: bool) -> dict[str, Any]:
    return {
        "type": "capabilities",
        "runtime": "qwen3-asr-mlx",
        "protocol": BRIDGE_PROTOCOL_VERSION,
        "model": ctx.model,
        "sample_rate": SAMPLE_RATE,
        "audio_format": "mono pcm16 little-endian base64 or WAV file path",
        "vad_boundary_source": "external",
        "use_cache": use_cache,
        "decoder_modes": ["sequential", "batched"] if use_cache else ["no_cache"],
        "messages": [
            "capabilities",
            "start",
            "transcribe",
            "batch_transcribe",
            "start_stream",
            "audio_chunk",
            "end_utterance",
            "vad_boundary",
            "flush",
            "stop_stream",
            "stop",
        ],
        "models": {
            "efficiency": "Qwen/Qwen3-ASR-0.6B",
            "accuracy_candidate": "Qwen/Qwen3-ASR-1.7B",
        },
    }


def memory_telemetry() -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss_peak = int(usage.ru_maxrss)
    if sys.platform != "darwin":
        rss_peak *= 1024
    telemetry: dict[str, Any] = {
        "rss_peak_bytes": rss_peak,
        "rss_peak_mb": round(rss_peak / 1_000_000, 2),
    }
    try:
        import mlx.core as mx

        for key, name in (
            ("mlx_peak_bytes", "get_peak_memory"),
            ("mlx_active_bytes", "get_active_memory"),
            ("mlx_cache_bytes", "get_cache_memory"),
        ):
            if hasattr(mx, name):
                value = int(getattr(mx, name)())
                telemetry[key] = value
                telemetry[key.replace("_bytes", "_mb")] = round(value / 1_000_000, 2)
    except Exception:
        pass
    return telemetry


def get_feat_extract_output_lengths(input_lengths: np.ndarray | int) -> np.ndarray:
    """Match vLLM/Qwen3-ASR _get_feat_extract_output_lengths."""
    lengths = np.asarray(input_lengths, dtype=np.int64)
    input_lengths_leave = lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (lengths // 100) * 13
    return output_lengths.astype(np.int64)


def import_qwen_asr_registration() -> None:
    """Import qwen-asr so AutoConfig/AutoProcessor know qwen3_asr classes."""
    try:
        import qwen_asr  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on runtime env
        raise ProbeError(
            "Could not import qwen_asr. Prepare the qwen3-asr-mlx runtime first. "
            "This probe expects qwen-asr without the [vllm] extra."
        ) from exc


def snapshot_or_local_model(ctx: ProbeContext) -> str:
    path = Path(ctx.model).expanduser()
    if path.exists():
        return str(path)

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # pragma: no cover - depends on runtime env
        raise ProbeError("huggingface_hub is required to locate/download the model") from exc

    try:
        return snapshot_download(
            repo_id=ctx.model,
            cache_dir=str(ctx.cache_dir),
            local_files_only=ctx.local_files_only,
        )
    except Exception as exc:
        mode = "local cache" if ctx.local_files_only else "Hugging Face download"
        raise ProbeError(f"Could not resolve {ctx.model} from {mode}: {exc}") from exc


def load_audio(ctx: ProbeContext) -> np.ndarray:
    if not ctx.audio_path:
        # Deterministic short tone: good for shape probes, not a recognition test.
        seconds = 1.0
        t = np.arange(int(seconds * SAMPLE_RATE), dtype=np.float32) / SAMPLE_RATE
        wav = 0.05 * np.sin(2 * np.pi * 440.0 * t)
        log("audio.synthetic", sample_rate=SAMPLE_RATE, samples=int(wav.shape[0]))
        return wav.astype(np.float32)

    try:
        import librosa
        import soundfile as sf
    except Exception as exc:  # pragma: no cover - depends on runtime env
        raise ProbeError("soundfile and librosa are required for audio loading") from exc

    wav, sr = sf.read(ctx.audio_path, dtype="float32", always_2d=False)
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim == 2:
        wav = wav.mean(axis=1 if wav.shape[0] > wav.shape[1] else 0).astype(np.float32)
    if int(sr) != SAMPLE_RATE:
        wav = librosa.resample(wav, orig_sr=int(sr), target_sr=SAMPLE_RATE).astype(np.float32)
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak > 1.0:
        wav = wav / peak
    wav = np.clip(wav, -1.0, 1.0).astype(np.float32)
    log("audio.loaded", path=ctx.audio_path, sample_rate=SAMPLE_RATE, samples=int(wav.shape[0]))
    return wav


def load_config_processor(ctx: ProbeContext, model_dir: str):
    import_qwen_asr_registration()
    from transformers import AutoConfig, AutoProcessor

    config = AutoConfig.from_pretrained(
        model_dir,
        trust_remote_code=ctx.trust_remote_code,
        local_files_only=ctx.local_files_only,
    )
    processor = AutoProcessor.from_pretrained(
        model_dir,
        trust_remote_code=ctx.trust_remote_code,
        local_files_only=ctx.local_files_only,
        fix_mistral_regex=True,
    )
    return config, processor


def token_summary(processor: Any) -> dict[str, Any]:
    tokenizer = processor.tokenizer
    tokens = {
        "audio_token": getattr(processor, "audio_token", "<|audio_pad|>"),
        "audio_bos_token": getattr(processor, "audio_bos_token", "<|audio_start|>"),
        "audio_eos_token": getattr(processor, "audio_eos_token", "<|audio_end|>"),
        "eos_token": getattr(tokenizer, "eos_token", None),
        "pad_token": getattr(tokenizer, "pad_token", None),
    }
    ids: dict[str, int | None] = {}
    for name, tok in tokens.items():
        ids[name] = tokenizer.convert_tokens_to_ids(tok) if tok is not None else None
    return {"tokens": tokens, "ids": ids}


def build_prompt(processor: Any, language: str | None) -> str:
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": [{"type": "audio", "audio": ""}]},
    ]
    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    if language:
        prompt += f"language {language}{ASR_TEXT_TAG}"
    return prompt


def processor_inputs(processor: Any, prompt: str, wav: np.ndarray) -> dict[str, np.ndarray]:
    batch = processor(
        text=prompt,
        audio=[wav],
        return_tensors="np",
        padding=True,
    )
    return {k: np.asarray(v) for k, v in batch.items()}


def processor_batch_inputs(processor: Any, prompt: str, wavs: list[np.ndarray]) -> dict[str, np.ndarray]:
    batch = processor(
        text=[prompt] * len(wavs),
        audio=wavs,
        return_tensors="np",
        padding=True,
    )
    return {k: np.asarray(v) for k, v in batch.items()}


def processor_audio_inputs(processor: Any, wavs: list[np.ndarray]) -> dict[str, np.ndarray]:
    from qwen_asr.core.transformers_backend.processing_qwen3_asr import Qwen3ASRProcessorKwargs

    output_kwargs = processor._merge_kwargs(
        Qwen3ASRProcessorKwargs,
        tokenizer_init_kwargs=processor.tokenizer.init_kwargs,
        return_tensors="np",
        padding=True,
    )
    audio_kwargs = dict(output_kwargs["audio_kwargs"])
    audio_kwargs["padding"] = True
    audio_kwargs["truncation"] = False
    audio_inputs = processor.feature_extractor(wavs, **audio_kwargs)
    return {
        "input_features": np.asarray(audio_inputs["input_features"]),
        "feature_attention_mask": np.asarray(audio_inputs["attention_mask"]),
    }


def make_prompt_token_template(processor: Any, prompt: str, audio_pad_id: int) -> PromptTokenTemplate:
    tokenized = processor.tokenizer(prompt, return_tensors="np", padding=False)
    ids = np.asarray(tokenized["input_ids"], dtype=np.int32).reshape(-1)
    pad_positions = np.flatnonzero(ids == int(audio_pad_id))
    if pad_positions.size != 1:
        raise ProbeError(
            f"Prompt template expected exactly one audio placeholder, found {pad_positions.size}"
        )
    pad_position = int(pad_positions[0])
    return PromptTokenTemplate(
        prefix=ids[:pad_position].copy(),
        suffix=ids[pad_position + 1 :].copy(),
        audio_pad_id=int(audio_pad_id),
    )


def assemble_prompt_input_ids(template: PromptTokenTemplate, audio_output_length: int) -> np.ndarray:
    ids = np.concatenate(
        [
            template.prefix,
            np.full(int(audio_output_length), int(template.audio_pad_id), dtype=np.int32),
            template.suffix,
        ]
    )
    return ids.reshape(1, -1)


def mrope_positions_for_audio_span(seq_len: int, audio_pad_span: tuple[int, int]) -> np.ndarray:
    seq_len = int(seq_len)
    offset, audio_end = (int(audio_pad_span[0]), int(audio_pad_span[1]))
    audio_len = audio_end - offset
    if offset < 0 or audio_len <= 0 or audio_end > seq_len:
        raise ProbeError(
            f"Invalid audio span {audio_pad_span} for sequence length {seq_len}"
        )
    pieces: list[np.ndarray] = []
    st = 0

    def add_positions(length: int) -> None:
        if length <= 0:
            return
        start = int(pieces[-1].max() + 1) if pieces else 0
        pos = np.arange(length, dtype=np.int64).reshape(1, -1)
        pieces.append(np.repeat(pos, 3, axis=0) + start)

    add_positions(offset - st)
    add_positions(audio_len)
    st = audio_end
    add_positions(seq_len - st)

    positions = np.concatenate(pieces, axis=1) if pieces else np.zeros((3, 0), dtype=np.int64)
    if positions.shape != (3, seq_len):
        raise ProbeError(f"MRoPE shape mismatch: got {positions.shape}, expected {(3, seq_len)}")
    return positions


def mrope_positions(input_ids: np.ndarray, audio_pad_id: int, audio_len: int) -> np.ndarray:
    ids = np.asarray(input_ids, dtype=np.int64).reshape(-1)
    span = audio_pad_token_span(ids, int(audio_pad_id), int(audio_len))
    return mrope_positions_for_audio_span(int(ids.shape[0]), span)


def config_to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "to_dict"):
        return dict(value.to_dict())
    if isinstance(value, dict):
        return dict(value)
    return dict(getattr(value, "__dict__", {}))


def mapped_text_weight_name(key: str) -> str | None:
    if key.startswith("thinker.model."):
        return key[len("thinker.") :]
    if key.startswith("thinker.lm_head."):
        return key[len("thinker.") :]
    return None


def interleaved_mrope_cos_sin(position_ids: np.ndarray, text_config: dict[str, Any]):
    import mlx.core as mx

    head_dim = int(text_config.get("head_dim") or text_config["hidden_size"] // text_config["num_attention_heads"])
    rope_theta = float(text_config.get("rope_theta", 1_000_000.0))
    rope_scaling = text_config.get("rope_scaling") or {}
    mrope_section = list(rope_scaling.get("mrope_section", [24, 20, 20]))
    if sum(int(v) for v in mrope_section) != head_dim // 2:
        raise ProbeError(
            f"mrope_section sum {sum(mrope_section)} does not match head_dim//2 {head_dim // 2}"
        )

    positions = np.asarray(position_ids, dtype=np.float32)
    if positions.ndim == 2:
        positions = positions[:, None, :]
    if positions.shape[0] != 3:
        raise ProbeError(f"MRoPE positions must have first dimension 3, got {positions.shape}")

    pos = mx.array(positions, dtype=mx.float32)
    inv_freq = 1.0 / (rope_theta ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim))
    freqs = (inv_freq[None, None, :, None] * pos[:, :, None, :]).transpose(0, 1, 3, 2)
    source_dims = [0] * (head_dim // 2)
    for dim, offset in enumerate((1, 2), start=1):
        length = int(mrope_section[dim]) * 3
        for idx in range(offset, length, 3):
            source_dims[idx] = dim
    freqs_t = mx.concatenate(
        [freqs[source_dim, ..., idx : idx + 1] for idx, source_dim in enumerate(source_dims)],
        axis=-1,
    )
    emb = mx.concatenate([freqs_t, freqs_t], axis=-1)
    cos = mx.cos(emb)
    sin = mx.sin(emb)
    mx.eval(cos, sin)
    return cos, sin


def load_mapped_qwen3_text_model(model_dir: str, text_config: dict[str, Any]):
    import mlx.core as mx
    from mlx_lm.models import qwen3

    files = sorted(glob.glob(str(Path(model_dir) / "*.safetensors")))
    if not files:
        files = sorted(glob.glob(str(Path(model_dir) / "**" / "*.safetensors"), recursive=True))
    if not files:
        raise ProbeError(f"No safetensors found in {model_dir}")

    text_config = dict(text_config)
    text_config["model_type"] = "qwen3"
    args = qwen3.ModelArgs.from_dict(text_config)
    model = qwen3.Model(args)

    weights: dict[str, Any] = {}
    for file in files:
        for key, value in mx.load(file).items():
            mapped = mapped_text_weight_name(key)
            if mapped is not None:
                weights[mapped] = value
    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    model.eval()
    return model


def mlx_rotate_half(x: Any) -> Any:
    import mlx.core as mx

    half = x.shape[-1] // 2
    return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def apply_mrope_to_qk(q: Any, k: Any, cos: Any, sin: Any) -> tuple[Any, Any]:
    import mlx.core as mx

    cos = mx.expand_dims(cos, axis=1)
    sin = mx.expand_dims(sin, axis=1)
    return (q * cos) + (mlx_rotate_half(q) * sin), (k * cos) + (mlx_rotate_half(k) * sin)


def make_qwen3_asr_mrope_model_class():
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models import qwen3
    from mlx_lm.models.base import create_attention_mask, scaled_dot_product_attention

    class MRoPEAttention(nn.Module):
        def __init__(self, args: qwen3.ModelArgs):
            super().__init__()
            dim = args.hidden_size
            self.n_heads = n_heads = args.num_attention_heads
            self.n_kv_heads = n_kv_heads = args.num_key_value_heads
            head_dim = args.head_dim
            self.scale = head_dim**-0.5
            self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
            self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
            self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
            self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)
            self.q_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
            self.k_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)

        def __call__(self, x: Any, position_embeddings: tuple[Any, Any], mask: Any = None, cache: Any = None) -> Any:
            bsz, length, _ = x.shape
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
            q = self.q_norm(q.reshape(bsz, length, self.n_heads, -1)).transpose(0, 2, 1, 3)
            k = self.k_norm(k.reshape(bsz, length, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
            v = v.reshape(bsz, length, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
            q, k = apply_mrope_to_qk(q, k, *position_embeddings)
            if cache is not None:
                k, v = cache.update_and_fetch(k, v)
            out = scaled_dot_product_attention(q, k, v, cache=cache, scale=self.scale, mask=mask)
            return self.o_proj(out.transpose(0, 2, 1, 3).reshape(bsz, length, -1))

    class MRoPETransformerBlock(nn.Module):
        def __init__(self, args: qwen3.ModelArgs):
            super().__init__()
            self.self_attn = MRoPEAttention(args)
            self.mlp = qwen3.MLP(args.hidden_size, args.intermediate_size)
            self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        def __call__(self, x: Any, position_embeddings: tuple[Any, Any], mask: Any = None, cache: Any = None) -> Any:
            h = x + self.self_attn(self.input_layernorm(x), position_embeddings, mask, cache)
            return h + self.mlp(self.post_attention_layernorm(h))

    class MRoPEQwen3Model(nn.Module):
        def __init__(self, args: qwen3.ModelArgs):
            super().__init__()
            self.args = args
            self.vocab_size = args.vocab_size
            self.num_hidden_layers = args.num_hidden_layers
            self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
            self.layers = [MRoPETransformerBlock(args) for _ in range(args.num_hidden_layers)]
            self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        def __call__(
            self,
            inputs: Any,
            position_embeddings: tuple[Any, Any],
            cache: Any = None,
            input_embeddings: Any = None,
        ) -> Any:
            h = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)
            if cache is None:
                cache = [None] * len(self.layers)
            mask = create_attention_mask(h, cache[0])
            for layer, layer_cache in zip(self.layers, cache):
                h = layer(h, position_embeddings, mask, layer_cache)
            return self.norm(h)

    class MRoPEQwen3ForCausalLM(nn.Module):
        def __init__(self, args: qwen3.ModelArgs):
            super().__init__()
            self.args = args
            self.model_type = args.model_type
            self.model = MRoPEQwen3Model(args)
            if not args.tie_word_embeddings:
                self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

        def __call__(
            self,
            inputs: Any,
            position_embeddings: tuple[Any, Any],
            cache: Any = None,
            input_embeddings: Any = None,
        ) -> Any:
            out = self.model(inputs, position_embeddings, cache, input_embeddings)
            if self.args.tie_word_embeddings:
                return self.model.embed_tokens.as_linear(out)
            return self.lm_head(out)

        def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
            if self.args.tie_word_embeddings:
                weights.pop("lm_head.weight", None)
            return weights

        @property
        def layers(self) -> list[Any]:
            return self.model.layers

    return MRoPEQwen3ForCausalLM


def load_mapped_qwen3_asr_mrope_model(model_dir: str, text_config: dict[str, Any]):
    import mlx.core as mx
    from mlx_lm.models import qwen3

    files = sorted(glob.glob(str(Path(model_dir) / "*.safetensors")))
    if not files:
        files = sorted(glob.glob(str(Path(model_dir) / "**" / "*.safetensors"), recursive=True))
    if not files:
        raise ProbeError(f"No safetensors found in {model_dir}")

    text_config = dict(text_config)
    text_config["model_type"] = "qwen3"
    args = qwen3.ModelArgs.from_dict(text_config)
    model_class = make_qwen3_asr_mrope_model_class()
    model = model_class(args)

    weights: dict[str, Any] = {}
    for file in files:
        for key, value in mx.load(file).items():
            mapped = mapped_text_weight_name(key)
            if mapped is not None:
                weights[mapped] = value
    weights = model.sanitize(weights)
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    model.eval()
    return model


def build_sinusoidal_positions(length: int, channels: int, max_timescale: int = 10000):
    import mlx.core as mx

    if channels % 2 != 0:
        raise ProbeError("Sinusoidal position embedding requires an even channel count")
    log_increment = math.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = mx.exp(-log_increment * mx.arange(channels // 2, dtype=mx.float32))
    scaled_time = mx.arange(length, dtype=mx.float32)[:, None] * inv_timescales[None, :]
    return mx.concatenate([mx.sin(scaled_time), mx.cos(scaled_time)], axis=1)


def make_audio_attention_mask(cu_seqlens: list[int], dtype: Any):
    import mlx.core as mx

    total = int(cu_seqlens[-1])
    mask = np.full((total, total), -1e9, dtype=np.float32)
    for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:]):
        mask[int(start) : int(end), int(start) : int(end)] = 0.0
    return mx.array(mask, dtype=dtype)[None, None, :, :]


def make_qwen3_asr_audio_tower_class():
    import mlx.core as mx
    import mlx.nn as nn

    class AudioAttention(nn.Module):
        def __init__(self, config: dict[str, Any]):
            super().__init__()
            self.embed_dim = int(config["d_model"])
            self.num_heads = int(config["encoder_attention_heads"])
            self.head_dim = self.embed_dim // self.num_heads
            self.scale = self.head_dim**-0.5
            self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
            self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
            self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
            self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

        def __call__(self, x: Any, attention_mask: Any = None) -> Any:
            length = x.shape[0]
            q = self.q_proj(x).reshape(length, self.num_heads, self.head_dim).transpose(1, 0, 2)[None]
            k = self.k_proj(x).reshape(length, self.num_heads, self.head_dim).transpose(1, 0, 2)[None]
            v = self.v_proj(x).reshape(length, self.num_heads, self.head_dim).transpose(1, 0, 2)[None]
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=attention_mask)
            return self.out_proj(out.reshape(self.num_heads, length, self.head_dim).transpose(1, 0, 2).reshape(length, -1))

    class AudioEncoderLayer(nn.Module):
        def __init__(self, config: dict[str, Any]):
            super().__init__()
            embed_dim = int(config["d_model"])
            self.self_attn = AudioAttention(config)
            self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
            self.fc1 = nn.Linear(embed_dim, int(config["encoder_ffn_dim"]))
            self.fc2 = nn.Linear(int(config["encoder_ffn_dim"]), embed_dim)
            self.final_layer_norm = nn.LayerNorm(embed_dim)

        def __call__(self, x: Any, attention_mask: Any = None) -> Any:
            x = x + self.self_attn(self.self_attn_layer_norm(x), attention_mask)
            return x + self.fc2(nn.gelu(self.fc1(self.final_layer_norm(x))))

    class AudioTower(nn.Module):
        def __init__(self, config: dict[str, Any]):
            super().__init__()
            self.config = config
            self.num_mel_bins = int(config["num_mel_bins"])
            self.d_model = int(config["d_model"])
            self.max_source_positions = int(config["max_source_positions"])
            self.n_window = int(config["n_window"])
            self.n_window_infer = int(config["n_window_infer"])
            self.downsample_hidden_size = int(config["downsample_hidden_size"])
            self.conv_chunksize = int(config.get("conv_chunksize", 500))
            self.layers = [AudioEncoderLayer(config) for _ in range(int(config["encoder_layers"]))]
            self.ln_post = nn.LayerNorm(self.d_model)
            self.conv2d1 = nn.Conv2d(1, self.downsample_hidden_size, 3, 2, padding=1)
            self.conv2d2 = nn.Conv2d(self.downsample_hidden_size, self.downsample_hidden_size, 3, 2, padding=1)
            self.conv2d3 = nn.Conv2d(self.downsample_hidden_size, self.downsample_hidden_size, 3, 2, padding=1)
            conv_freq = (((self.num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2
            self.conv_out = nn.Linear(self.downsample_hidden_size * conv_freq, self.d_model, bias=False)
            self.proj1 = nn.Linear(self.d_model, self.d_model)
            self.proj2 = nn.Linear(self.d_model, int(config["output_dim"]))
            self._sinusoidal_cache: dict[tuple[int, str], Any] = {}
            self._attention_mask_cache: dict[tuple[tuple[int, ...], str], Any] = {}

        def sinusoidal_positions(self, length: int, dtype: Any) -> Any:
            key = (int(length), str(dtype))
            cached = self._sinusoidal_cache.get(key)
            if cached is None:
                cached = build_sinusoidal_positions(length, self.d_model).astype(dtype)
                mx.eval(cached)
                self._sinusoidal_cache[key] = cached
            return cached

        def audio_attention_mask(self, cu_seqlens: list[int], dtype: Any) -> Any:
            key = (tuple(int(value) for value in cu_seqlens), str(dtype))
            cached = self._attention_mask_cache.get(key)
            if cached is None:
                cached = make_audio_attention_mask(cu_seqlens, dtype)
                mx.eval(cached)
                self._attention_mask_cache[key] = cached
            return cached

        def __call__(self, input_features: Any, feature_lens: np.ndarray) -> Any:
            feature_lens = np.asarray(feature_lens, dtype=np.int64).reshape(-1)
            chunks: list[Any] = []
            chunk_lengths: list[int] = []
            max_chunk = self.n_window * 2
            for batch_idx, feature_len in enumerate(feature_lens):
                total = int(feature_len)
                start = 0
                while start < total:
                    length = min(max_chunk, total - start)
                    chunk = input_features[batch_idx, :, start : start + length].T
                    chunks.append(chunk)
                    chunk_lengths.append(length)
                    start += length
            if not chunks:
                raise ProbeError("Audio tower received no feature chunks")

            padded_length = max(chunk_lengths)
            padded = []
            for chunk, length in zip(chunks, chunk_lengths):
                if length < padded_length:
                    pad = mx.zeros((padded_length - length, self.num_mel_bins), dtype=chunk.dtype)
                    chunk = mx.concatenate([chunk, pad], axis=0)
                padded.append(chunk.T)
            x = mx.stack(padded, axis=0)
            x = x[..., None]
            x = nn.gelu(self.conv2d1(x))
            x = nn.gelu(self.conv2d2(x))
            x = nn.gelu(self.conv2d3(x))
            aftercnn_lengths = get_feat_extract_output_lengths(np.asarray(chunk_lengths, dtype=np.int64))
            batch_chunks, freq, time, channels = x.shape
            x = x.transpose(0, 2, 3, 1).reshape(batch_chunks, time, channels * freq)
            x = self.conv_out(x)
            x = x + self.sinusoidal_positions(int(x.shape[1]), x.dtype)[None]
            hidden_parts = [x[idx, : int(length)] for idx, length in enumerate(aftercnn_lengths)]
            hidden = mx.concatenate(hidden_parts, axis=0)

            full_aftercnn_lens = get_feat_extract_output_lengths(feature_lens)
            window_aftercnn = int(x.shape[1]) * (self.n_window_infer // (self.n_window * 2))
            cu = [0]
            for length in full_aftercnn_lens:
                remaining = int(length)
                while remaining >= window_aftercnn and window_aftercnn > 0:
                    cu.append(cu[-1] + window_aftercnn)
                    remaining -= window_aftercnn
                if remaining:
                    cu.append(cu[-1] + remaining)
            if cu[-1] != int(hidden.shape[0]):
                cu = [0, int(hidden.shape[0])]
            attention_mask = self.audio_attention_mask(cu, hidden.dtype)
            for layer in self.layers:
                hidden = layer(hidden, attention_mask)
            hidden = self.ln_post(hidden)
            hidden = self.proj1(hidden)
            hidden = nn.gelu(hidden)
            return self.proj2(hidden)

    return AudioTower


def load_mapped_audio_tower(model_dir: str, audio_config: dict[str, Any]):
    import mlx.core as mx

    files = sorted(glob.glob(str(Path(model_dir) / "*.safetensors")))
    if not files:
        files = sorted(glob.glob(str(Path(model_dir) / "**" / "*.safetensors"), recursive=True))
    if not files:
        raise ProbeError(f"No safetensors found in {model_dir}")

    model_class = make_qwen3_asr_audio_tower_class()
    model = model_class(audio_config)
    weights: dict[str, Any] = {}
    for file in files:
        for key, value in mx.load(file).items():
            if not key.startswith("thinker.audio_tower."):
                continue
            mapped = key[len("thinker.audio_tower.") :]
            if mapped.startswith("conv2d") and mapped.endswith(".weight"):
                value = value.transpose(0, 2, 3, 1)
            weights[mapped] = value
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    model.eval()
    return model


def audio_pad_token_span(input_ids: Any, audio_pad_id: int, audio_embeddings_len: int) -> tuple[int, int]:
    ids = np.asarray(input_ids).reshape(-1)
    pad_positions = np.flatnonzero(ids == int(audio_pad_id))
    if pad_positions.size != int(audio_embeddings_len):
        raise ProbeError(
            f"Audio splice found {pad_positions.size} placeholder tokens, but audio tower produced {audio_embeddings_len}"
        )
    if pad_positions.size == 0:
        raise ProbeError("Audio splice found no placeholder tokens")
    if not np.all(np.diff(pad_positions) == 1):
        raise ProbeError("Audio placeholder tokens are not contiguous; cannot use fast splice path")
    return int(pad_positions[0]), int(pad_positions[-1]) + 1


def splice_audio_embeddings(
    model: Any,
    input_ids: Any,
    audio_pad_id: int,
    audio_embeddings: Any,
    audio_pad_span: tuple[int, int] | None = None,
) -> Any:
    import mlx.core as mx

    text_embeddings = model.model.embed_tokens(input_ids)
    start, end = audio_pad_span or audio_pad_token_span(
        np.asarray(input_ids.tolist()).reshape(-1),
        audio_pad_id,
        int(audio_embeddings.shape[0]),
    )
    return mx.concatenate(
        [
            text_embeddings[:, :start, :],
            audio_embeddings[None, :, :],
            text_embeddings[:, end:, :],
        ],
        axis=1,
    )


class Qwen3ASRMLXRuntime:
    """Reusable MLX runtime independent of app wiring."""

    def __init__(self, ctx: ProbeContext):
        import mlx.core as mx

        self.ctx = ctx
        self.model_dir = snapshot_or_local_model(ctx)
        self.config, self.processor = load_config_processor(ctx, self.model_dir)
        thinker = getattr(self.config, "thinker_config", None)
        self.text_config = config_to_dict(getattr(thinker, "text_config", None))
        self.audio_config = config_to_dict(getattr(thinker, "audio_config", None))
        self.token_info = token_summary(self.processor)
        self.audio_pad_id = self.token_info["ids"]["audio_token"]
        self.eos_token_id = self.token_info["ids"]["eos_token"]
        if self.audio_pad_id is None:
            raise ProbeError("Processor/tokenizer did not expose an audio token id")
        self.prompt = build_prompt(self.processor, ctx.language)
        self.prompt_token_template = make_prompt_token_template(
            self.processor,
            self.prompt,
            int(self.audio_pad_id),
        )
        self.audio_tower = load_mapped_audio_tower(self.model_dir, self.audio_config)
        self.decoder = load_mapped_qwen3_asr_mrope_model(self.model_dir, self.text_config)
        self._mrope_step_cache: dict[tuple[int, int, int], tuple[Any, Any]] = {}
        self.last_profile: dict[str, Any] = {}
        self._init_mrope_cache()

    def _init_mrope_cache(self) -> None:
        import mlx.core as mx

        head_dim = int(self.text_config.get("head_dim") or self.text_config["hidden_size"] // self.text_config["num_attention_heads"])
        rope_theta = float(self.text_config.get("rope_theta", 1_000_000.0))
        rope_scaling = self.text_config.get("rope_scaling") or {}
        self.mrope_section = list(rope_scaling.get("mrope_section", [24, 20, 20]))
        self.mrope_inv_freq = 1.0 / (rope_theta ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim))
        source_dims = [0] * (head_dim // 2)
        for dim, offset in enumerate((1, 2), start=1):
            length = int(self.mrope_section[dim]) * 3
            for idx in range(offset, length, 3):
                source_dims[idx] = dim
        self.mrope_source_dims = source_dims
        mx.eval(self.mrope_inv_freq)

    def mrope_cos_sin(self, position_ids: np.ndarray):
        import mlx.core as mx

        positions = np.asarray(position_ids, dtype=np.float32)
        if positions.ndim == 2:
            positions = positions[:, None, :]
        cache_key = None
        if positions.shape == (3, 1, 1):
            cache_key = tuple(int(value) for value in positions[:, 0, 0])
            cached = self._mrope_step_cache.get(cache_key)
            if cached is not None:
                return cached
        pos = mx.array(positions, dtype=mx.float32)
        freqs = (self.mrope_inv_freq[None, None, :, None] * pos[:, :, None, :]).transpose(0, 1, 3, 2)
        freqs_t = mx.concatenate(
            [freqs[source_dim, ..., idx : idx + 1] for idx, source_dim in enumerate(self.mrope_source_dims)],
            axis=-1,
        )
        emb = mx.concatenate([freqs_t, freqs_t], axis=-1)
        cos = mx.cos(emb)
        sin = mx.sin(emb)
        mx.eval(cos, sin)
        if cache_key is not None:
            self._mrope_step_cache[cache_key] = (cos, sin)
        return cos, sin

    def mrope_cos_sin_for_text_positions(self, positions: list[int]):
        position_ids = np.asarray(
            [
                positions,
                positions,
                positions,
            ],
            dtype=np.int64,
        )[:, :, None]
        return self.mrope_cos_sin(position_ids)

    def prepare(self, wav: np.ndarray) -> dict[str, Any]:
        import mlx.core as mx

        total_start = time.perf_counter()
        feature_start = time.perf_counter()
        inputs = processor_audio_inputs(self.processor, [wav])
        feature_seconds = time.perf_counter() - feature_start
        feature_mask = inputs["feature_attention_mask"]
        audio_feature_lengths = feature_mask.sum(axis=-1).astype(np.int64)
        audio_output_lengths = get_feat_extract_output_lengths(audio_feature_lengths)
        prompt_start = time.perf_counter()
        inputs["input_ids"] = assemble_prompt_input_ids(self.prompt_token_template, int(audio_output_lengths[0]))
        inputs["attention_mask"] = np.ones_like(inputs["input_ids"], dtype=np.int32)
        prompt_seconds = time.perf_counter() - prompt_start
        input_features = mx.array(inputs["input_features"])
        tower_start = time.perf_counter()
        audio_embeddings = self.audio_tower(input_features, audio_feature_lengths)
        mx.eval(audio_embeddings)
        audio_tower_seconds = time.perf_counter() - tower_start
        self.last_profile = {
            "stage": "prepare",
            "batch_size": 1,
            "audio_samples": [int(wav.shape[0])],
            "audio_seconds": round(float(wav.shape[0]) / SAMPLE_RATE, 4),
            "feature_extract_seconds": round(feature_seconds, 4),
            "prompt_assembly_seconds": round(prompt_seconds, 4),
            "audio_tower_seconds": round(audio_tower_seconds, 4),
            "total_seconds": round(time.perf_counter() - total_start, 4),
            "audio_feature_lengths": audio_feature_lengths.astype(int).tolist(),
            "audio_output_lengths": audio_output_lengths.astype(int).tolist(),
            "input_features_shape": list(inputs["input_features"].shape),
            "audio_embeddings_shape": list(audio_embeddings.shape),
            "memory": memory_telemetry(),
        }
        return {
            "inputs": inputs,
            "audio_feature_lengths": audio_feature_lengths,
            "audio_output_lengths": audio_output_lengths,
            "audio_embeddings": audio_embeddings,
        }

    def prepare_batch(self, wavs: list[np.ndarray]) -> list[dict[str, Any]]:
        import mlx.core as mx

        if not wavs:
            return []
        total_start = time.perf_counter()
        feature_start = time.perf_counter()
        inputs = processor_audio_inputs(self.processor, wavs)
        feature_seconds = time.perf_counter() - feature_start
        feature_mask = inputs["feature_attention_mask"]
        audio_feature_lengths = feature_mask.sum(axis=-1).astype(np.int64)
        audio_output_lengths = get_feat_extract_output_lengths(audio_feature_lengths)
        tower_start = time.perf_counter()
        audio_embeddings = self.audio_tower(mx.array(inputs["input_features"]), audio_feature_lengths)
        mx.eval(audio_embeddings)
        audio_tower_seconds = time.perf_counter() - tower_start

        items: list[dict[str, Any]] = []
        offset = 0
        prompt_start = time.perf_counter()
        for idx, output_length in enumerate(audio_output_lengths):
            length = int(output_length)
            ids = assemble_prompt_input_ids(self.prompt_token_template, length).reshape(-1)
            items.append(
                {
                    "input_ids": ids.astype(np.int32).reshape(1, -1),
                    "audio_feature_length": int(audio_feature_lengths[idx]),
                    "audio_output_length": length,
                    "audio_embeddings": audio_embeddings[offset : offset + length],
                }
            )
            offset += length
        if offset != int(audio_embeddings.shape[0]):
            raise ProbeError(f"Batch audio split consumed {offset}, tower produced {audio_embeddings.shape[0]}")
        prompt_seconds = time.perf_counter() - prompt_start
        audio_samples = [int(wav.shape[0]) for wav in wavs]
        self.last_profile = {
            "stage": "prepare_batch",
            "batch_size": len(wavs),
            "audio_samples": audio_samples,
            "audio_seconds": round(sum(audio_samples) / SAMPLE_RATE, 4),
            "feature_extract_seconds": round(feature_seconds, 4),
            "prompt_assembly_seconds": round(prompt_seconds, 4),
            "audio_tower_seconds": round(audio_tower_seconds, 4),
            "total_seconds": round(time.perf_counter() - total_start, 4),
            "audio_feature_lengths": audio_feature_lengths.astype(int).tolist(),
            "audio_output_lengths": audio_output_lengths.astype(int).tolist(),
            "input_features_shape": list(inputs["input_features"].shape),
            "audio_embeddings_shape": list(audio_embeddings.shape),
            "memory": memory_telemetry(),
        }
        return items

    def decode_prepared_item_cached(
        self,
        input_ids_np: np.ndarray,
        audio_embeddings: Any,
        audio_output_length: int,
        max_new_tokens: int | None = None,
        ) -> dict[str, Any]:
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache

        limit = max_new_tokens if max_new_tokens is not None else self.ctx.max_new_tokens
        max_steps = int(limit) if limit is not None else 1024
        decode_start = time.perf_counter()
        cache, generated, next_position_value = self.prefill_prepared_item(
            input_ids_np,
            audio_embeddings,
            audio_output_length,
        )

        for _ in range(max(0, max_steps - 1)):
            if self.eos_token_id is not None and generated[-1] == int(self.eos_token_id):
                break
            positions = np.full((3, 1), next_position_value, dtype=np.int64)
            cos, sin = self.mrope_cos_sin(positions)
            step_input = mx.array([[generated[-1]]], dtype=mx.int32)
            logits = self.decoder(step_input, (cos, sin), cache=cache)
            next_token = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(next_token)
            token_id = int(np.asarray(next_token)[0])
            generated.append(token_id)
            next_position_value += 1

        decode_seconds = time.perf_counter() - decode_start
        return {
            "generated_token_ids": generated,
            "text": self.processor.tokenizer.decode(generated, skip_special_tokens=True),
            "audio_embeddings_shape": list(audio_embeddings.shape),
            "audio_output_lengths": [int(audio_output_length)],
            "decode_seconds": decode_seconds,
            "cache_final_offset": int(cache[0].offset) if cache else 0,
        }

    def prefill_prepared_item(
        self,
        input_ids_np: np.ndarray,
        audio_embeddings: Any,
        audio_output_length: int,
    ) -> tuple[list[Any], list[int], int]:
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache

        cache = [KVCache() for _ in self.decoder.layers]

        audio_pad_span = audio_pad_token_span(input_ids_np[0], int(self.audio_pad_id), int(audio_output_length))
        positions = mrope_positions_for_audio_span(input_ids_np.shape[1], audio_pad_span)
        cos, sin = self.mrope_cos_sin(positions)
        next_position_value = int(positions[0, -1]) + 1 if positions.shape[1] else 0
        input_ids = mx.array(input_ids_np)
        spliced = splice_audio_embeddings(
            self.decoder,
            input_ids,
            int(self.audio_pad_id),
            audio_embeddings,
            audio_pad_span=audio_pad_span,
        )
        logits = self.decoder(input_ids, (cos, sin), cache=cache, input_embeddings=spliced)
        next_token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(next_token)
        token_id = int(np.asarray(next_token)[0])
        return cache, [token_id], next_position_value

    def transcribe_array(self, wav: np.ndarray, max_new_tokens: int | None = None) -> dict[str, Any]:
        import mlx.core as mx

        limit = max_new_tokens if max_new_tokens is not None else self.ctx.max_new_tokens
        max_steps = int(limit) if limit is not None else 1024
        prepared = self.prepare(wav)
        input_ids_np = prepared["inputs"]["input_ids"].astype(np.int32)
        audio_output_lengths = prepared["audio_output_lengths"]
        audio_embeddings = prepared["audio_embeddings"]
        audio_pad_span = audio_pad_token_span(
            input_ids_np[0],
            int(self.audio_pad_id),
            int(audio_output_lengths[0]),
        )
        generated: list[int] = []
        decode_start = time.perf_counter()
        for _ in range(max(1, max_steps)):
            positions = mrope_positions_for_audio_span(input_ids_np.shape[1], audio_pad_span)
            cos, sin = self.mrope_cos_sin(positions)
            input_ids = mx.array(input_ids_np)
            spliced = splice_audio_embeddings(
                self.decoder,
                input_ids,
                int(self.audio_pad_id),
                audio_embeddings,
                audio_pad_span=audio_pad_span,
            )
            logits = self.decoder(input_ids, (cos, sin), input_embeddings=spliced)
            next_token = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(next_token)
            token_id = int(np.asarray(next_token)[0])
            generated.append(token_id)
            input_ids_np = np.concatenate([input_ids_np, np.asarray([[token_id]], dtype=np.int32)], axis=1)
            if self.eos_token_id is not None and token_id == int(self.eos_token_id):
                break
        decode_seconds = time.perf_counter() - decode_start
        return {
            "generated_token_ids": generated,
            "text": self.processor.tokenizer.decode(generated, skip_special_tokens=True),
            "audio_embeddings_shape": list(audio_embeddings.shape),
            "audio_output_lengths": audio_output_lengths.tolist(),
            "decode_seconds": decode_seconds,
        }

    def transcribe_array_cached(self, wav: np.ndarray, max_new_tokens: int | None = None) -> dict[str, Any]:
        total_start = time.perf_counter()
        prepared = self.prepare(wav)
        prepare_profile = dict(self.last_profile)
        input_ids_np = prepared["inputs"]["input_ids"].astype(np.int32)
        output = self.decode_prepared_item_cached(
            input_ids_np,
            prepared["audio_embeddings"],
            int(prepared["audio_output_lengths"][0]),
            max_new_tokens=max_new_tokens,
        )
        total_seconds = time.perf_counter() - total_start
        generated_tokens = len(output["generated_token_ids"])
        self.last_profile = {
            "stage": "cached_transcribe",
            "batch_size": 1,
            "audio_seconds": prepare_profile.get("audio_seconds"),
            "prepare": prepare_profile,
            "decode_seconds": round(float(output["decode_seconds"]), 4),
            "total_seconds": round(total_seconds, 4),
            "generated_tokens": [generated_tokens],
            "generated_tokens_total": generated_tokens,
            "generated_tokens_per_second": round(generated_tokens / float(output["decode_seconds"]), 2)
            if float(output["decode_seconds"]) > 0
            else None,
            "realtime_factor": round(float(prepare_profile.get("audio_seconds") or 0.0) / total_seconds, 2) if total_seconds > 0 else None,
            "memory": memory_telemetry(),
        }
        return output

    def transcribe_batch_cached(self, wavs: list[np.ndarray], max_new_tokens: int | None = None) -> list[dict[str, Any]]:
        total_start = time.perf_counter()
        prepared_items = self.prepare_batch(wavs)
        prepare_profile = dict(self.last_profile)
        decode_start = time.perf_counter()
        outputs = [
            self.decode_prepared_item_cached(
                item["input_ids"],
                item["audio_embeddings"],
                int(item["audio_output_length"]),
                max_new_tokens=max_new_tokens,
            )
            for item in prepared_items
        ]
        decode_seconds = time.perf_counter() - decode_start
        total_seconds = time.perf_counter() - total_start
        generated_token_counts = [len(output["generated_token_ids"]) for output in outputs]
        generated_tokens_total = int(sum(generated_token_counts))
        self.last_profile = {
            "stage": "batch_cached_sequential",
            "batch_size": len(prepared_items),
            "audio_seconds": prepare_profile.get("audio_seconds"),
            "prepare": prepare_profile,
            "decode_seconds": round(decode_seconds, 4),
            "total_seconds": round(total_seconds, 4),
            "generated_tokens": generated_token_counts,
            "generated_tokens_total": generated_tokens_total,
            "generated_tokens_per_second": round(generated_tokens_total / decode_seconds, 2) if decode_seconds > 0 else None,
            "realtime_factor": round(float(prepare_profile.get("audio_seconds") or 0.0) / total_seconds, 2) if total_seconds > 0 else None,
            "memory": memory_telemetry(),
        }
        return outputs

    def transcribe_batch_batched_decode(
        self,
        wavs: list[np.ndarray],
        max_new_tokens: int | None = None,
    ) -> list[dict[str, Any]]:
        import mlx.core as mx
        from mlx_lm.models.cache import BatchKVCache

        total_start = time.perf_counter()
        prepared_items = self.prepare_batch(wavs)
        prepare_profile = dict(self.last_profile)
        if not prepared_items:
            self.last_profile = {
                "stage": "batch_batched_decode",
                "batch_size": 0,
                "total_seconds": 0.0,
                "memory": memory_telemetry(),
            }
            return []

        limit = max_new_tokens if max_new_tokens is not None else self.ctx.max_new_tokens
        max_steps = int(limit) if limit is not None else 1024
        decode_start = time.perf_counter()

        generated: list[list[int]] = [[] for _ in prepared_items]
        next_positions: list[int] = []
        finished = [False] * len(prepared_items)
        lengths = [int(item["input_ids"].shape[1]) for item in prepared_items]
        max_len = max(lengths)
        left_padding = [max_len - length for length in lengths]
        hidden_size = int(self.text_config["hidden_size"])

        padded_ids: list[Any] = []
        padded_embeddings: list[Any] = []
        padded_positions: list[np.ndarray] = []
        pad_token_id = int(self.token_info["ids"].get("pad_token") or 0)

        assembly_start = time.perf_counter()
        for item, length, pad in zip(prepared_items, lengths, left_padding):
            input_ids_np = item["input_ids"].astype(np.int32)
            audio_output_length = int(item["audio_output_length"])
            audio_span = audio_pad_token_span(input_ids_np[0], int(self.audio_pad_id), audio_output_length)
            input_ids = mx.array(input_ids_np)
            embeddings = splice_audio_embeddings(
                self.decoder,
                input_ids,
                int(self.audio_pad_id),
                item["audio_embeddings"],
                audio_pad_span=audio_span,
            )[0]
            positions = mrope_positions_for_audio_span(length, audio_span)

            if pad > 0:
                padded_ids.append(
                    mx.concatenate(
                        [
                            mx.full((pad,), pad_token_id, dtype=mx.int32),
                            input_ids[0],
                        ],
                        axis=0,
                    )
                )
                padded_embeddings.append(
                    mx.concatenate(
                        [
                            mx.zeros((pad, hidden_size), dtype=embeddings.dtype),
                            embeddings,
                        ],
                        axis=0,
                    )
                )
                padded_positions.append(
                    np.concatenate(
                        [
                            np.zeros((3, pad), dtype=np.int64),
                            positions,
                        ],
                        axis=1,
                    )
                )
            else:
                padded_ids.append(input_ids[0])
                padded_embeddings.append(embeddings)
                padded_positions.append(positions)
            next_positions.append(int(positions[0, -1]) + 1)

        batch_ids = mx.stack(padded_ids, axis=0)
        batch_embeddings = mx.stack(padded_embeddings, axis=0)
        batch_positions = np.stack(padded_positions, axis=1)
        batch_cache = [BatchKVCache(left_padding) for _ in self.decoder.layers]
        batch_assembly_seconds = time.perf_counter() - assembly_start

        prefill_seconds = 0.0
        if max_len > 1:
            prefill_start = time.perf_counter()
            prefill_positions = batch_positions[:, :, :-1]
            cos, sin = self.mrope_cos_sin(prefill_positions)
            self.decoder(
                batch_ids[:, :-1],
                (cos, sin),
                cache=batch_cache,
                input_embeddings=batch_embeddings[:, :-1, :],
            )
            mx.eval([cache.state for cache in batch_cache])
            prefill_seconds = time.perf_counter() - prefill_start

        active_items = list(range(len(prepared_items)))
        step_input = batch_ids[:, -1:]
        step_positions = batch_positions[:, :, -1:]
        continuation_start = time.perf_counter()
        for _ in range(max(1, max_steps)):
            if isinstance(step_positions, list):
                cos, sin = self.mrope_cos_sin_for_text_positions(step_positions)
            else:
                cos, sin = self.mrope_cos_sin(step_positions)
            logits = self.decoder(step_input, (cos, sin), cache=batch_cache)
            next_tokens = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(next_tokens)

            token_ids = np.asarray(next_tokens).astype(np.int64).reshape(-1).tolist()
            keep_positions: list[int] = []
            keep_items: list[int] = []
            next_step_tokens: list[list[int]] = []
            next_step_positions: list[int] = []
            for active_pos, item_idx in enumerate(active_items):
                token_id = int(token_ids[active_pos])
                generated[item_idx].append(token_id)
                next_positions[item_idx] += 1
                if self.eos_token_id is not None and token_id == int(self.eos_token_id):
                    finished[item_idx] = True
                else:
                    keep_positions.append(active_pos)
                    keep_items.append(item_idx)
                    next_step_tokens.append([token_id])
                    next_step_positions.append(next_positions[item_idx])

            if not keep_positions:
                break
            if len(keep_positions) != len(active_items):
                for cache in batch_cache:
                    cache.filter(keep_positions)
            active_items = keep_items
            step_input = mx.array(next_step_tokens, dtype=mx.int32)
            step_positions = next_step_positions

        continuation_seconds = time.perf_counter() - continuation_start
        decode_seconds = time.perf_counter() - decode_start
        per_item_decode = decode_seconds / max(1, len(prepared_items))
        generated_token_counts = [len(item_generated) for item_generated in generated]
        generated_tokens_total = int(sum(generated_token_counts))
        total_seconds = time.perf_counter() - total_start
        self.last_profile = {
            "stage": "batch_batched_decode",
            "batch_size": len(prepared_items),
            "audio_seconds": prepare_profile.get("audio_seconds"),
            "prepare": prepare_profile,
            "batch_assembly_seconds": round(batch_assembly_seconds, 4),
            "prefill_seconds": round(prefill_seconds, 4),
            "continuation_seconds": round(continuation_seconds, 4),
            "decode_seconds": round(decode_seconds, 4),
            "total_seconds": round(total_seconds, 4),
            "generated_tokens": generated_token_counts,
            "generated_tokens_total": generated_tokens_total,
            "generated_tokens_per_second": round(generated_tokens_total / decode_seconds, 2) if decode_seconds > 0 else None,
            "realtime_factor": round(float(prepare_profile.get("audio_seconds") or 0.0) / total_seconds, 2) if total_seconds > 0 else None,
            "prompt_lengths": lengths,
            "max_prompt_length": max_len,
            "left_padding": left_padding,
            "memory": memory_telemetry(),
        }
        return [
            {
                "generated_token_ids": item_generated,
                "text": self.processor.tokenizer.decode(item_generated, skip_special_tokens=True),
                "audio_embeddings_shape": list(item["audio_embeddings"].shape),
                "audio_output_lengths": [int(item["audio_output_length"])],
                "decode_seconds": per_item_decode,
                "cache_final_offset": None,
            }
            for idx, (item, item_generated) in enumerate(zip(prepared_items, generated))
        ]

    def transcribe_file(self, audio_path: str, max_new_tokens: int | None = None) -> dict[str, Any]:
        file_ctx = dataclasses.replace(self.ctx, audio_path=audio_path)
        wav = load_audio(file_ctx)
        return self.transcribe_array(wav, max_new_tokens=max_new_tokens)

    def transcribe_file_cached(self, audio_path: str, max_new_tokens: int | None = None) -> dict[str, Any]:
        file_ctx = dataclasses.replace(self.ctx, audio_path=audio_path)
        wav = load_audio(file_ctx)
        return self.transcribe_array_cached(wav, max_new_tokens=max_new_tokens)


def message_token_limit(message: dict[str, Any], default: int | None) -> int | None:
    value = message.get("max_new_tokens", default)
    if value is None:
        return None
    value_int = int(value)
    return None if value_int == 0 else value_int


def message_decoder_mode(message: dict[str, Any]) -> str:
    mode = str(message.get("decoder_mode", "sequential")).lower()
    if mode not in {"batched", "sequential"}:
        raise ProbeError(f"decoder_mode must be 'batched' or 'sequential', got {mode!r}")
    return mode


def pcm16_base64_to_float32(value: str) -> np.ndarray:
    raw = base64.b64decode(value)
    pcm = np.frombuffer(raw, dtype="<i2")
    return (pcm.astype(np.float32) / 32768.0).clip(-1.0, 1.0)


@dataclasses.dataclass
class RealtimeStreamState:
    stream_id: str
    sample_rate: int = SAMPLE_RATE
    batch_size: int = 4
    max_batch_delay_ms: int = 120
    audio: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    chunk_count: int = 0
    utterance_count: int = 0


@dataclasses.dataclass
class QueuedUtterance:
    stream_id: str
    utterance_id: str
    audio: np.ndarray
    samples: int
    source: str


class RuntimeJSONBridge:
    def __init__(self, ctx: ProbeContext, use_cache: bool) -> None:
        self.ctx = ctx
        self.use_cache = use_cache
        self.runtime: Qwen3ASRMLXRuntime | None = None
        self.streams: dict[str, RealtimeStreamState] = {}
        self.utterance_queue: list[QueuedUtterance] = []

    def ensure_runtime(self) -> tuple[Qwen3ASRMLXRuntime, float | None]:
        if self.runtime is not None:
            return self.runtime, None
        started = time.perf_counter()
        self.runtime = Qwen3ASRMLXRuntime(self.ctx)
        return self.runtime, time.perf_counter() - started

    def stream_state(self, message: dict[str, Any]) -> RealtimeStreamState:
        stream_id = str(message.get("stream_id") or "default")
        state = self.streams.get(stream_id)
        if state is None:
            state = RealtimeStreamState(stream_id=stream_id)
            self.streams[stream_id] = state
        return state

    def decode_message_audio(self, message: dict[str, Any], state: RealtimeStreamState) -> tuple[np.ndarray, str]:
        if "audio" in message:
            path = str(message["audio"])
            return load_audio(dataclasses.replace(self.ctx, audio_path=path)), path
        if "pcm16" in message:
            sample_rate = int(message.get("sample_rate", state.sample_rate))
            if sample_rate != SAMPLE_RATE:
                raise ProbeError(f"Realtime bridge expects {SAMPLE_RATE} Hz PCM, got {sample_rate}")
            return pcm16_base64_to_float32(str(message.get("pcm16", ""))), "pcm16"
        if state.audio.shape[0] == 0:
            raise ProbeError("No buffered audio for utterance boundary")
        return state.audio.copy(), "buffer"

    def queue_utterance(self, message: dict[str, Any]) -> dict[str, Any]:
        state = self.stream_state(message)
        wav, source = self.decode_message_audio(message, state)
        state.utterance_count += 1
        utterance_id = str(message.get("utterance_id") or f"{state.stream_id}-{state.utterance_count}")
        self.utterance_queue.append(
            QueuedUtterance(
                stream_id=state.stream_id,
                utterance_id=utterance_id,
                audio=wav.astype(np.float32, copy=False),
                samples=int(wav.shape[0]),
                source=source,
            )
        )
        state.audio = np.zeros((0,), dtype=np.float32)

        if len(self.utterance_queue) >= max(1, state.batch_size):
            return self.flush_realtime(message, reason="batch_size")
        return {
            "type": "utterance_queued",
            "stream_id": state.stream_id,
            "utterance_id": utterance_id,
            "samples": int(wav.shape[0]),
            "queued": len(self.utterance_queue),
            "batch_size": state.batch_size,
        }

    def flush_realtime(self, message: dict[str, Any], reason: str = "flush") -> dict[str, Any]:
        runtime, load_seconds = self.ensure_runtime()
        limit = message_token_limit(message, self.ctx.max_new_tokens)
        if not self.utterance_queue:
            return {
                "type": "realtime_batch_final",
                "reason": reason,
                "count": 0,
                "items": [],
                "load_seconds": round(load_seconds, 4) if load_seconds is not None else 0.0,
                "use_cache": self.use_cache,
            }

        batch = self.utterance_queue
        self.utterance_queue = []
        started = time.perf_counter()
        decoder_mode = message_decoder_mode(message)
        with contextlib.redirect_stdout(sys.stderr):
            wavs = [item.audio for item in batch]
            if self.use_cache and decoder_mode == "batched":
                outputs = runtime.transcribe_batch_batched_decode(wavs, max_new_tokens=limit)
            elif self.use_cache:
                outputs = runtime.transcribe_batch_cached(wavs, max_new_tokens=limit)
            else:
                outputs = [runtime.transcribe_array(wav, max_new_tokens=limit) for wav in wavs]
        return {
            "type": "realtime_batch_final",
            "reason": reason,
            "count": len(batch),
            "items": [
                {
                    "type": "final_result",
                    "stream_id": item.stream_id,
                    "utterance_id": item.utterance_id,
                    "source": item.source,
                    "samples": item.samples,
                    "text": output["text"],
                    "generated_tokens": len(output["generated_token_ids"]),
                    "audio_embeddings_shape": output["audio_embeddings_shape"],
                    "decode_seconds": round(float(output["decode_seconds"]), 4),
                }
                for item, output in zip(batch, outputs)
            ],
            "total_seconds": round(time.perf_counter() - started, 4),
            "load_seconds": round(load_seconds, 4) if load_seconds is not None else 0.0,
            "use_cache": self.use_cache,
            "decoder_mode": decoder_mode if self.use_cache else "no_cache",
            "asr_backend": "mlx",
            "vad_backend": "external-boundary",
            "profile": runtime.last_profile,
        }

    def handle(self, message: dict[str, Any]) -> dict[str, Any]:
        msg_type = message.get("type")
        if msg_type in {"capabilities", "hello"}:
            return bridge_capabilities(self.ctx, self.use_cache)
        if msg_type == "start":
            runtime, load_seconds = self.ensure_runtime()
            return {
                "type": "ready",
                "runtime": "qwen3-asr-mlx",
                "protocol": BRIDGE_PROTOCOL_VERSION,
                "model": runtime.ctx.model,
                "model_dir": runtime.model_dir,
                "use_cache": self.use_cache,
                "load_seconds": round(load_seconds, 4) if load_seconds is not None else 0.0,
            }
        if msg_type == "start_stream":
            self.ensure_runtime()
            stream_id = str(message.get("stream_id") or "default")
            state = RealtimeStreamState(
                stream_id=stream_id,
                sample_rate=int(message.get("sample_rate", SAMPLE_RATE)),
                batch_size=max(1, int(message.get("batch_size", 4))),
                max_batch_delay_ms=max(0, int(message.get("max_batch_delay_ms", 120))),
            )
            if state.sample_rate != SAMPLE_RATE:
                return {
                    "type": "error",
                    "message": f"Realtime bridge expects {SAMPLE_RATE} Hz PCM, got {state.sample_rate}",
                }
            self.streams[stream_id] = state
            return {
                "type": "stream_ready",
                "protocol": BRIDGE_PROTOCOL_VERSION,
                "stream_id": stream_id,
                "sample_rate": state.sample_rate,
                "batch_size": state.batch_size,
                "max_batch_delay_ms": state.max_batch_delay_ms,
                "asr_backend": "mlx",
                "vad_backend": "external-boundary",
                "note": "Send VAD-finalized boundaries with end_utterance or vad_boundary; use flush for timer-based micro-batches.",
            }
        if msg_type == "audio_chunk":
            state = self.stream_state(message)
            sample_rate = int(message.get("sample_rate", state.sample_rate))
            if sample_rate != SAMPLE_RATE:
                return {"type": "error", "message": f"Realtime bridge expects {SAMPLE_RATE} Hz PCM, got {sample_rate}"}
            chunk = pcm16_base64_to_float32(str(message.get("pcm16", "")))
            state.audio = np.concatenate([state.audio, chunk])
            state.chunk_count += 1
            return {
                "type": "partial_result",
                "stream_id": state.stream_id,
                "text": "",
                "samples_buffered": int(state.audio.shape[0]),
                "chunks": state.chunk_count,
                "is_final": False,
            }
        if msg_type in {"end_utterance", "vad_boundary"}:
            return self.queue_utterance(message)
        if msg_type == "flush":
            return self.flush_realtime(message)
        if msg_type == "stop_stream":
            state = self.stream_state(message)
            dropped_samples = int(state.audio.shape[0])
            self.streams.pop(state.stream_id, None)
            return {
                "type": "stream_stopped",
                "stream_id": state.stream_id,
                "dropped_samples": dropped_samples,
                "queued": len(self.utterance_queue),
            }
        if msg_type == "transcribe":
            runtime, load_seconds = self.ensure_runtime()
            audio_path = str(message["audio"])
            limit = message_token_limit(message, self.ctx.max_new_tokens)
            started = time.perf_counter()
            with contextlib.redirect_stdout(sys.stderr):
                if self.use_cache:
                    output = runtime.transcribe_file_cached(audio_path, max_new_tokens=limit)
                else:
                    output = runtime.transcribe_file(audio_path, max_new_tokens=limit)
            return {
                "type": "transcript",
                "audio": audio_path,
                "text": output["text"],
                "generated_token_ids": output["generated_token_ids"],
                "generated_tokens": len(output["generated_token_ids"]),
                "audio_embeddings_shape": output["audio_embeddings_shape"],
                "audio_output_lengths": output["audio_output_lengths"],
                "decode_seconds": round(float(output["decode_seconds"]), 4),
                "total_seconds": round(time.perf_counter() - started, 4),
                "load_seconds": round(load_seconds, 4) if load_seconds is not None else 0.0,
                "use_cache": self.use_cache,
                "profile": runtime.last_profile,
            }
        if msg_type == "batch_transcribe":
            runtime, load_seconds = self.ensure_runtime()
            audio_paths = [str(path) for path in message.get("audio", [])]
            if not audio_paths:
                return {"type": "error", "message": "batch_transcribe requires a non-empty audio list"}
            limit = message_token_limit(message, self.ctx.max_new_tokens)
            decoder_mode = message_decoder_mode(message)
            started = time.perf_counter()
            with contextlib.redirect_stdout(sys.stderr):
                wavs = [load_audio(dataclasses.replace(self.ctx, audio_path=path)) for path in audio_paths]
                if self.use_cache and decoder_mode == "batched":
                    outputs = runtime.transcribe_batch_batched_decode(wavs, max_new_tokens=limit)
                elif self.use_cache:
                    outputs = runtime.transcribe_batch_cached(wavs, max_new_tokens=limit)
                else:
                    outputs = [runtime.transcribe_array(wav, max_new_tokens=limit) for wav in wavs]
            return {
                "type": "batch_transcript",
                "count": len(audio_paths),
                "items": [
                    {
                        "audio": path,
                        "text": output["text"],
                        "generated_tokens": len(output["generated_token_ids"]),
                        "audio_embeddings_shape": output["audio_embeddings_shape"],
                        "decode_seconds": round(float(output["decode_seconds"]), 4),
                    }
                    for path, output in zip(audio_paths, outputs)
                ],
                "total_seconds": round(time.perf_counter() - started, 4),
                "load_seconds": round(load_seconds, 4) if load_seconds is not None else 0.0,
                "use_cache": self.use_cache,
                "decoder_mode": decoder_mode if self.use_cache else "no_cache",
                "profile": runtime.last_profile,
            }
        if msg_type == "stop":
            return {"type": "stopped"}
        return {"type": "error", "message": f"unknown message type: {msg_type}"}


def run_runtime_bridge(ctx: ProbeContext, use_cache: bool) -> None:
    global LOG_STREAM
    LOG_STREAM = sys.stderr
    bridge = RuntimeJSONBridge(ctx, use_cache=use_cache)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            response = bridge.handle(json.loads(line))
        except Exception as exc:
            response = {"type": "error", "message": str(exc), "traceback": traceback.format_exc()}
        print(json.dumps(response, ensure_ascii=False), flush=True)
        if response.get("type") == "stopped":
            break
