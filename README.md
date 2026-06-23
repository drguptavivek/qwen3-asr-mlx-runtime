# qwen3-asr-mlx-runtime

Apple Silicon MLX runtime and newline-JSON bridge for `Qwen/Qwen3-ASR-0.6B`
and `Qwen/Qwen3-ASR-1.7B`.

This repository is a code-only runtime. It does not include Qwen model weights,
private audio, or app-specific UI code. The goal is to make Qwen3-ASR usable as
a local worker process from Swift, Node/web backends, Python apps, and command
line tools.

## Status

- License: Apache-2.0
- Runtime protocol: `qwen3-asr-mlx-jsonl-v1`
- Platform: macOS on Apple Silicon
- Python: native arm64 Python 3.12
- Backend: MLX + MLX-LM
- VAD: external boundary source
- Default model: `Qwen/Qwen3-ASR-0.6B`
- Accuracy candidate: `Qwen/Qwen3-ASR-1.7B`

This is an experimental runtime. The current implementation focuses on local
correctness, module boundaries, and reusable app integration. It is not an
official Qwen project.

## Why This Exists

The official Qwen3-ASR package supports Transformers and vLLM backends. This
runtime provides a separate Apple Silicon path using MLX:

- Qwen3-ASR audio tower in MLX
- MRoPE-aware Qwen decoder wrapper
- audio embedding splice into text embeddings
- cached greedy decode
- VAD-sized batch decode
- newline-delimited JSON bridge for non-Python apps

## Requirements

- Apple Silicon Mac
- macOS with Metal available
- Native arm64 Python 3.12, or `uv`
- Xcode Command Line Tools recommended

Install `uv` if you want the launcher to manage Python automatically:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Quick Setup

Clone the repo:

```bash
git clone https://github.com/drguptavivek/qwen3-asr-mlx-runtime.git
cd qwen3-asr-mlx-runtime
```

Start the bridge with the default efficiency model:

```bash
scripts/qwen3-asr-mlx-bridge --local-files-only
```

If the model is not cached yet, omit `--local-files-only` for the first run:

```bash
scripts/qwen3-asr-mlx-bridge
```

Run the heavier accuracy candidate:

```bash
scripts/qwen3-asr-mlx-bridge Qwen/Qwen3-ASR-1.7B
```

Print capabilities without loading the model:

```bash
scripts/qwen3-asr-mlx-bridge --print-capabilities
```

## Basic JSONL Use

The bridge reads one JSON object per line from stdin and writes one JSON object
per line to stdout. Diagnostic logs go to stderr.

```bash
printf '%s\n' \
  '{"type":"start"}' \
  '{"type":"transcribe","audio":"examples/audio/sample.wav","max_new_tokens":0}' \
  '{"type":"stop"}' \
| scripts/qwen3-asr-mlx-bridge --local-files-only
```

Response shape:

```json
{"type":"transcript","audio":"examples/audio/sample.wav","text":"language English<asr_text>...","generated_tokens":13,"audio_embeddings_shape":[32,1024],"decode_seconds":0.4,"total_seconds":0.6,"profile":{}}
```

## App Integration

Use the bridge as a local worker process. Keep the model resident and send
requests over stdin/stdout.

Supported request types:

- `capabilities`
- `start`
- `transcribe`
- `batch_transcribe`
- `start_stream`
- `audio_chunk`
- `end_utterance`
- `vad_boundary`
- `flush`
- `stop_stream`
- `stop`

The bridge expects VAD boundaries from the caller. A Swift app can use its
native VAD stack. A web backend can use a VAD worker and forward finalized
utterances to this bridge.

See:

- [Protocol](docs/protocol.md)
- [Architecture](docs/architecture.md)
- [Performance](docs/performance.md)
- [Submodule integration](docs/submodule.md)
- [Swift example](examples/swift/SubprocessBridge.swift)
- [Node example](examples/node/bridge.mjs)
- [Python example](examples/python/client.py)

## Model Choice

Use `Qwen/Qwen3-ASR-0.6B` for realtime/live transcription by default. Use
`Qwen/Qwen3-ASR-1.7B` as an opt-in accuracy candidate when the user accepts
higher latency and memory use.

The 1.7B model produced cleaner wording on our small smoke set, but it is not
proven more accurate until evaluated on labeled audio with WER/CER.

## Performance Snapshot

Measured on three VAD-sized saved-session WAVs totaling `10.1001s` of audio.
These are runtime smoke numbers, not a formal benchmark.

| Model | Intended role | Decoder mode | Total time | Realtime factor | Peak RSS | MLX peak |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `Qwen/Qwen3-ASR-0.6B` | realtime efficiency default | `sequential` | `1.1157s` | `9.06x` | `2418.84 MB` | `3093.95 MB` |
| `Qwen/Qwen3-ASR-0.6B` | lowest-latency draft/speed mode | `batched` | `0.6936s` | `14.59x` | `2409.76 MB` | `3179.04 MB` |
| `Qwen/Qwen3-ASR-1.7B` | accuracy candidate | `sequential` | `2.1896s` | `4.61x` | `4914.04 MB` | `6609.91 MB` |
| `Qwen/Qwen3-ASR-1.7B` | heavier batched candidate | `batched` | `1.4309s` | `7.06x` | `4927.28 MB` | `6663.12 MB` |

Observed transcript differences on the smoke set:

| Segment | 0.6B sequential | 1.7B sequential | Assessment |
| --- | --- | --- | --- |
| `segment_6_input.wav` | `Sorry.` | `Sorry.` | Equivalent. |
| `segment_2_input.wav` | `Chris tries there. Hey, Chris.` | `Chris Drysdale.` | 1.7B is shorter and more name-like, but needs ground truth before calling it correct. |
| `segment_1_input.wav` | `Possibly, maybe he wanna hang out with the cool guy.` | `Possibly, maybe he'd want to hang out with the cool guy.` | 1.7B is more grammatical; both preserve the same meaning. |

## Decoder Modes

- `sequential`: batches feature extraction and the MLX audio tower, then decodes
  each segment with its own cached decoder. Use this for final transcripts.
- `batched`: batches feature extraction, audio tower, multimodal prefill, and
  continuation decode with `BatchKVCache`. Faster, but small wording drift is
  possible when logits are nearly tied.

## Use As A Git Submodule

In a parent app repository:

```bash
git submodule add https://github.com/drguptavivek/qwen3-asr-mlx-runtime.git Vendor/qwen3-asr-mlx-runtime
git submodule update --init --recursive
```

Launch from the parent app:

```bash
Vendor/qwen3-asr-mlx-runtime/scripts/qwen3-asr-mlx-bridge --local-files-only
```

Pin updates explicitly:

```bash
cd Vendor/qwen3-asr-mlx-runtime
git fetch origin
git checkout <tag-or-commit>
cd ../..
git add Vendor/qwen3-asr-mlx-runtime
git commit -m "Update Qwen3-ASR MLX runtime"
```

## Environment Variables

| Variable | Purpose | Default |
| --- | --- | --- |
| `QWEN3_ASR_MLX_MODEL` | Default model if no positional model is passed | `Qwen/Qwen3-ASR-0.6B` |
| `QWEN3_ASR_MLX_HOME` | Runtime virtualenv and uv cache root | `~/.audio-models/runtimes/qwen3-asr-mlx` |
| `QWEN3_ASR_MLX_CACHE_DIR` | Hugging Face model cache | `~/.audio-models` |
| `QWEN3_ASR_MLX_PYTHON` | Python version for managed venv | `3.12` |
| `UV_CACHE_DIR` | uv package cache | `$QWEN3_ASR_MLX_HOME/uv-cache` |
| `UV_PYTHON_INSTALL_DIR` | uv-managed Python install cache | `$QWEN3_ASR_MLX_HOME/uv-python` |

## Attribution

This project depends on the Qwen3-ASR model family and upstream tooling:

- Qwen3-ASR model cards: <https://huggingface.co/Qwen/Qwen3-ASR-0.6B>
- Qwen3-ASR upstream repo: <https://github.com/QwenLM/Qwen3-ASR>
- Qwen3-ASR technical report: <https://arxiv.org/abs/2601.21337>
- MLX: <https://github.com/ml-explore/mlx>
- MLX-LM: <https://github.com/ml-explore/mlx-lm>

Model weights are downloaded from their upstream providers under their own
licenses. This repository does not redistribute them.
