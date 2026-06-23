#!/usr/bin/env python3
"""Performance test CLI for one or more Qwen3-ASR WAV/audio files."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any

from .runtime import (
    DEFAULT_CACHE,
    DEFAULT_MODEL,
    SAMPLE_RATE,
    ProbeContext,
    Qwen3ASRMLXRuntime,
    load_audio,
    set_log_stream,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Qwen3-ASR MLX performance test on one or more audio files")
    parser.add_argument("audio", nargs="+", help="WAV/audio file path. Pass multiple paths for VAD-style batch testing.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF repo id or local model path")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE), help="Model cache directory")
    parser.add_argument("--language", default=None, help="Optional forced Qwen language name, e.g. English")
    parser.add_argument(
        "--decoder-mode",
        choices=["sequential", "batched"],
        default="sequential",
        help="Decode mode for multiple files. sequential is token-stable; batched is faster but can drift on close logits.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=0,
        help="Generated-token cap. Use 0 to decode until EOS.",
    )
    parser.add_argument("--local-files-only", action="store_true", help="Do not download missing model files")
    parser.add_argument("--no-trust-remote-code", action="store_true", help="Disable Transformers trust_remote_code")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only")
    parser.add_argument("--show-text", action="store_true", help="Print decoded text in table mode")
    return parser.parse_args()


def text_after_asr_tag(text: str) -> str:
    marker = "<asr_text>"
    if marker in text:
        return text.split(marker, 1)[1]
    return text


def compact_profile(profile: dict[str, Any]) -> dict[str, Any]:
    memory = profile.get("memory") or {}
    return {
        "stage": profile.get("stage"),
        "batch_size": profile.get("batch_size"),
        "audio_seconds": profile.get("audio_seconds"),
        "total_seconds": profile.get("total_seconds"),
        "decode_seconds": profile.get("decode_seconds"),
        "generated_tokens_total": profile.get("generated_tokens_total"),
        "generated_tokens_per_second": profile.get("generated_tokens_per_second"),
        "realtime_factor": profile.get("realtime_factor"),
        "rss_peak_mb": memory.get("rss_peak_mb"),
        "mlx_peak_mb": memory.get("mlx_peak_mb"),
        "mlx_active_mb": memory.get("mlx_active_mb"),
        "mlx_cache_mb": memory.get("mlx_cache_mb"),
    }


def run_perftest(args: argparse.Namespace) -> dict[str, Any]:
    set_log_stream(sys.stderr)
    max_new_tokens = None if args.max_new_tokens == 0 else args.max_new_tokens
    ctx = ProbeContext(
        model=args.model,
        cache_dir=Path(args.cache_dir).expanduser(),
        audio_path=None,
        language=args.language,
        trust_remote_code=not args.no_trust_remote_code,
        local_files_only=args.local_files_only,
        max_new_tokens=max_new_tokens,
    )

    started = time.perf_counter()
    with contextlib.redirect_stdout(sys.stderr):
        runtime = Qwen3ASRMLXRuntime(ctx)
    load_seconds = time.perf_counter() - started

    audio_paths = [str(Path(path).expanduser()) for path in args.audio]
    with contextlib.redirect_stdout(sys.stderr):
        wavs = [load_audio(dataclasses.replace(ctx, audio_path=path)) for path in audio_paths]

    audio_seconds = [round(float(wav.shape[0]) / SAMPLE_RATE, 4) for wav in wavs]
    run_started = time.perf_counter()
    with contextlib.redirect_stdout(sys.stderr):
        if args.decoder_mode == "batched":
            outputs = runtime.transcribe_batch_batched_decode(wavs, max_new_tokens=max_new_tokens)
        else:
            outputs = runtime.transcribe_batch_cached(wavs, max_new_tokens=max_new_tokens)
    run_seconds = time.perf_counter() - run_started
    total_seconds = time.perf_counter() - started
    total_audio_seconds = round(sum(audio_seconds), 4)

    items = []
    for path, seconds, output in zip(audio_paths, audio_seconds, outputs):
        text = output["text"]
        items.append(
            {
                "audio": path,
                "audio_seconds": seconds,
                "generated_tokens": len(output["generated_token_ids"]),
                "decode_seconds": round(float(output["decode_seconds"]), 4),
                "audio_embeddings_shape": output["audio_embeddings_shape"],
                "text": text,
                "transcript": text_after_asr_tag(text),
            }
        )

    return {
        "type": "perftest",
        "runtime": "qwen3-asr-mlx",
        "model": runtime.ctx.model,
        "model_dir": runtime.model_dir,
        "decoder_mode": args.decoder_mode,
        "audio_count": len(audio_paths),
        "audio_seconds": total_audio_seconds,
        "load_seconds": round(load_seconds, 4),
        "load_profile": runtime.load_profile,
        "run_seconds": round(run_seconds, 4),
        "total_seconds_with_load": round(total_seconds, 4),
        "realtime_factor": round(total_audio_seconds / run_seconds, 2) if run_seconds > 0 else None,
        "realtime_factor_with_load": round(total_audio_seconds / total_seconds, 2) if total_seconds > 0 else None,
        "items": items,
        "profile": runtime.last_profile,
        "summary": compact_profile(runtime.last_profile),
    }


def print_table(result: dict[str, Any], show_text: bool) -> None:
    print("Qwen3-ASR MLX perftest")
    print(f"  model: {result['model']}")
    print(f"  decoder_mode: {result['decoder_mode']}")
    print(f"  audio_count: {result['audio_count']}")
    print(f"  audio_seconds: {result['audio_seconds']}")
    print(f"  load_seconds: {result['load_seconds']}")
    print(f"  run_seconds: {result['run_seconds']}")
    print(f"  realtime_factor: {result['realtime_factor']}x")
    print(f"  realtime_factor_with_load: {result['realtime_factor_with_load']}x")
    summary = result.get("summary") or {}
    if summary:
        print("  memory:")
        print(f"    rss_peak_mb: {summary.get('rss_peak_mb')}")
        print(f"    mlx_peak_mb: {summary.get('mlx_peak_mb')}")
        print(f"    mlx_active_mb: {summary.get('mlx_active_mb')}")
        print(f"    mlx_cache_mb: {summary.get('mlx_cache_mb')}")
    print("")
    print("files:")
    for item in result["items"]:
        print(
            f"  - {item['audio']} | audio={item['audio_seconds']}s "
            f"tokens={item['generated_tokens']} decode={item['decode_seconds']}s "
            f"emb={item['audio_embeddings_shape']}"
        )
        if show_text:
            print(f"    text: {item['transcript']}")


def main() -> int:
    args = parse_args()
    result = run_perftest(args)
    if args.json:
        print(json.dumps(result, ensure_ascii=False), flush=True)
    else:
        print_table(result, show_text=args.show_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
