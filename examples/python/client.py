#!/usr/bin/env python3
"""Minimal Python subprocess client for qwen3-asr-mlx-runtime."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(os.environ.get("QWEN3_ASR_MLX_RUNTIME_ROOT", Path.cwd()))
    audio = sys.argv[1] if len(sys.argv) > 1 else "examples/audio/sample.wav"
    proc = subprocess.Popen(
        [
            str(repo_root / "scripts" / "qwen3-asr-mlx-bridge"),
            "Qwen/Qwen3-ASR-0.6B",
            "--local-files-only",
        ],
        cwd=repo_root,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None
    proc.stdin.write(json.dumps({"type": "start"}) + "\n")
    proc.stdin.write(json.dumps({"type": "transcribe", "audio": audio, "max_new_tokens": 0}) + "\n")
    proc.stdin.write(json.dumps({"type": "stop"}) + "\n")
    proc.stdin.flush()
    for line in proc.stdout:
        print(json.loads(line))
    return proc.wait()


if __name__ == "__main__":
    raise SystemExit(main())
