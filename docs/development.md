# Development

## Local Environment

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -e .
```

The tested MLX stack is:

```text
mlx==0.31.2
mlx-lm==0.29.1
transformers==4.57.6
qwen-asr==0.0.6
```

Or use the managed launcher:

```bash
scripts/qwen3-asr-mlx-bridge --print-capabilities
```

## Lightweight Checks

```bash
python -m py_compile qwen3_asr_mlx_runtime/runtime.py qwen3_asr_mlx_runtime/bridge.py
bash -n scripts/qwen3-asr-mlx-bridge
python -m qwen3_asr_mlx_runtime.bridge --print-capabilities
```

## Metal Check

The full ASR path requires Metal access. In headless or sandboxed macOS
sessions, MLX may fail with:

```text
[metal::load_device] No Metal device available
```

Run generation tests from a process that can access the Apple GPU.
