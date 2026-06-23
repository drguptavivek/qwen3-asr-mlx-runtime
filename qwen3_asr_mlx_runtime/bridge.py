#!/usr/bin/env python3
"""Stable app-facing Qwen3-ASR MLX newline-JSON bridge.

This entrypoint is for Swift, web backends, command-line tools, and other apps.
Diagnostics/probes remain in probe_qwen3_asr_mlx.py; this file only exposes the
runtime protocol.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .runtime import (
    DEFAULT_CACHE,
    DEFAULT_MODEL,
    ProbeContext,
    bridge_capabilities,
    run_runtime_bridge,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen3-ASR MLX as an app-facing JSONL bridge")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF repo id or local model path")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE), help="Model cache directory")
    parser.add_argument("--language", default=None, help="Optional forced Qwen language name, e.g. English")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=0,
        help="Default generated-token cap. Use 0 to decode until EOS.",
    )
    parser.add_argument("--local-files-only", action="store_true", help="Do not download missing model files")
    parser.add_argument("--no-trust-remote-code", action="store_true", help="Disable Transformers trust_remote_code")
    parser.add_argument("--no-cache", action="store_true", help="Disable cached decoder generation")
    parser.add_argument(
        "--print-capabilities",
        action="store_true",
        help="Print the JSONL bridge capabilities and exit without loading the model.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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
    use_cache = not args.no_cache
    if args.print_capabilities:
        print(json.dumps(bridge_capabilities(ctx, use_cache), ensure_ascii=False), flush=True)
        return 0
    run_runtime_bridge(ctx, use_cache=use_cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
